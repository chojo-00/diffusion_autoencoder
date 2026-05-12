import os
import copy
import argparse
import random
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

from logger import Logger
import distributed_util as dist_util
from diffae.runner import Runner as Diffae_Runner
# from diffae.runner_latent import Runner as LDM_Runner
from dataset import dataset
from diffae import ckpt_util
from guided_diffusion.script_util import create_gaussian_diffusion

import colored_traceback.always
from ipdb import set_trace as debug

RESULT_DIR = Path("results/diffae")

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def build_subset_per_gpu(opt, dataset, log):
    n_data = len(dataset)
    n_gpu  = opt.global_size
    n_dump = (n_data % n_gpu > 0) * (n_gpu - n_data % n_gpu)

    # create index for each gpu
    total_idx = np.concatenate([np.arange(n_data), np.zeros(n_dump)]).astype(int)
    idx_per_gpu = total_idx.reshape(-1, n_gpu)[:, opt.global_rank]
    log.info(f"[Dataset] Add {n_dump} data to the end to be devided by {n_gpu}. Total length={len(total_idx)}!")

    # build subset
    indices = idx_per_gpu.tolist()
    subset = Subset(dataset, indices)
    log.info(f"[Dataset] Built subset for gpu={opt.global_rank}! Now size={len(subset)}!")
    return subset

def collect_all_subset(sample, log):
    batch, *xdim = sample.shape
    gathered_samples = dist_util.all_gather(sample, log)
    gathered_samples = [sample.cpu() for sample in gathered_samples]
    # [batch, n_gpu, *xdim] --> [batch*n_gpu, *xdim]
    return torch.stack(gathered_samples, dim=1).reshape(-1, *xdim)

def build_partition(opt, full_dataset, log):
    n_samples = len(full_dataset)

    part_idx, n_part = [int(s) for s in opt.partition.split("_")]
    assert part_idx < n_part and part_idx >= 0
    assert n_samples % n_part == 0

    n_samples_per_part = n_samples // n_part
    start_idx = part_idx * n_samples_per_part
    end_idx = (part_idx+1) * n_samples_per_part

    indices = [i for i in range(start_idx, end_idx)]
    subset = Subset(full_dataset, indices)
    log.info(f"[Dataset] Built partition={opt.partition}, {start_idx}, {end_idx}! Now size={len(subset)}!")
    return subset

def build_val_dataset(opt, log):
    val_dataset = dataset.Dataset(opt, log, mode='test')

    # build partition
    if opt.partition is not None:
        val_dataset = build_partition(opt, val_dataset, log)
    return val_dataset

def get_recon_imgs_fn(opt, nfe):
    recon_imgs_fn = RESULT_DIR / opt.ckpt / "samples_nfe{}{}_iter{}{}".format(
        nfe, "_clip" if opt.clip_denoise else "", opt.load_itr, "_" + str(opt.ldm_load_itr) if opt.use_ldm else ""
    )
    os.makedirs(recon_imgs_fn, exist_ok=True)

    return recon_imgs_fn

def generate_style(opt, log, ldm_runner, ldm_ckpt_opt, cond=None, nfe=50):
    diffusion = create_gaussian_diffusion(steps=ldm_ckpt_opt.interval, noise_schedule=ldm_ckpt_opt.schedule_name, timestep_respacing=f"ddim{nfe}")
    z_style = diffusion.ddim_sample_loop(ldm_runner.net, (opt.batch_size, 512), cond=cond, clip_denoised=ldm_ckpt_opt.clip_denoise, progress=True)
    log.info("Generated style feature!")
    return z_style

def compute_batch(ckpt_opt, out):
    img, label, fpath = out
    x0 = img.detach().to(torch.float32)
    return x0, label, fpath

@torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")

    # get (default) ckpt option
    diffae_ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)
    nfe = opt.nfe or diffae_ckpt_opt.interval-1

    # build imagenet val dataset
    val_dataset = build_val_dataset(opt, log)
    n_samples = len(val_dataset)

    # build dataset per gpu and loader
    subset_dataset = build_subset_per_gpu(opt, val_dataset, log)
    val_loader = DataLoader(subset_dataset,
        batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
    )

    # build runner
    diffae_runner = Diffae_Runner(diffae_ckpt_opt, log, save_opt=False)

    # handle use_fp16 for ema
    if opt.use_fp16:
        diffae_runner.ema.copy_to() # copy weight from ema to net
        diffae_runner.net.diffusion_model.convert_to_fp16()
        diffae_runner.net.semantic_enc.convert_to_fp16()
        diffae_runner.ema = ExponentialMovingAverage(diffae_runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight

    # use ldm runner
    # if opt.use_ldm:
    #     ldm_ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / "latent-ddim" / opt.ldm_ckpt, net="latent-ddim")
    #     ldm_runner = LDM_Runner(ldm_ckpt_opt, log, save_opt=False)
    #     if opt.use_fp16:
    #         ldm_runner.ema.copy_to() # copy weight from ema to net
    #         ldm_runner.net.convert_to_fp16()
    #         ldm_runner.ema = ExponentialMovingAverage(ldm_runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight

    # create save folder
    recon_imgs_fn = get_recon_imgs_fn(opt, nfe)
    log.info(f"Recon images will be saved to {recon_imgs_fn}!")

    recon_imgs = []
    num = 0

    for loader_itr, out in enumerate(val_loader):
        x0, label, fpath = compute_batch(diffae_ckpt_opt, out)

        # generate style using latent ddim network
        # generated_style = generate_style(opt, log, ldm_runner=ldm_runner, ldm_ckpt_opt=ldm_ckpt_opt, cond=x0, nfe=nfe) if opt.use_ldm else None
        batch, *xdim = x0.shape
        recon_img = diffae_runner.ddpm_sampling(diffae_ckpt_opt, x0, batch, nfe=nfe)
        recon_img = recon_img.to(opt.device)
        if opt.clip_denoise: recon_img.clamp_(-1., 1.)

        assert recon_img.shape == x0.shape

        tu.save_image((x0+1)/2, recon_imgs_fn / f"{loader_itr:05}_target.png", value_range=(0, 1))
        tu.save_image((recon_img+1)/2, recon_imgs_fn / f"{loader_itr:05}_target_recon.png", value_range=(0, 1))            
        log.info("Saved output images!")

        # [-1,1]
        gathered_recon_img = collect_all_subset(recon_img, log)
        recon_imgs.append(gathered_recon_img)

        num += len(gathered_recon_img)
        log.info(f"Collected {num} recon images!")
        dist.barrier()

    del diffae_runner

    arr = torch.cat(recon_imgs, axis=0)[:n_samples]

    if opt.global_rank == 0:
        torch.save({"arr": arr}, recon_imgs_fn / "recon.pt")
        log.info(f"Save at {recon_imgs_fn}")
    dist.barrier()

    log.info(f"Sampling complete! Collect recon_imgs={arr.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default='localhost', help="address for master")
    parser.add_argument("--master-port",    type=str,  default='6020',      help="port for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")

    # data
    parser.add_argument("--image-size",     type=int,  default=256)
    parser.add_argument("--dataset-dir",    type=Path, default="/data",     help="path to dataset")
    parser.add_argument("--partition",      type=str,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")

    # latent ddim
    parser.add_argument("--ldm-load-itr",   type=int,  default=200000)
    parser.add_argument("--ldm-ckpt",       type=str,  default=None,        help="the checkpoint name from which we wish to sample from ldm")
    parser.add_argument("--use-ldm",        action="store_true",            help="use latent ddim network for generating semantic style")

    # sample
    parser.add_argument("--load-itr",       type=int,  default=80000)
    parser.add_argument("--batch-size",     type=int,  default=1)
    parser.add_argument("--ckpt",           type=str,  default=None,        help="the checkpoint name from which we wish to sample from diffae")
    parser.add_argument("--nfe",            type=int,  default=250,         help="sampling steps")
    parser.add_argument("--clip-denoise",   action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")

    arg = parser.parse_args()

    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device="cuda",
    )
    opt.update(vars(arg))

    set_seed(opt.seed)

    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=dist_util.init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        dist_util.init_processes(0, opt.n_gpu_per_node, main, opt)
