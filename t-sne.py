import os
import copy
import cv2
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.manifold import TSNE
from easydict import EasyDict as edict

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

from logger import Logger
import distributed_util as dist_util
from diffae.runner import Runner
from dataset.image_dataset import PNGGrayDataset
from diffae import ckpt_util

import colored_traceback.always
from ipdb import set_trace as debug

RESULT_DIR = Path("results")

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
    val_dataset = PNGGrayDataset(opt, log, mode='test')

    # build partition
    if opt.partition is not None:
        val_dataset = build_partition(opt, val_dataset, log)
    return val_dataset

def get_t_sne_fn(opt):
    t_sne_fn = RESULT_DIR / opt.ckpt / f"t-sne_iter{opt.load_itr}"
    os.makedirs(t_sne_fn, exist_ok=True)

    return t_sne_fn

def compute_batch(ckpt_opt, out):
    img, label, fpath = out
    x0 = img.detach().to(torch.float32)
    return x0, label, fpath

def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def draw_rectangle_by_class(image, label, palette):
    image_height, image_width, _ = image.shape

    colors = [(255,0,0),(0,255,0),(0,0,255), (0,125,125)]
    # get the color corresponding to image class
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=colors[label], thickness=5)

    return image

def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y

@torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")

    # get (default) ckpt option
    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)

    # build imagenet val dataset
    val_dataset = build_val_dataset(opt, log)
    n_samples = len(val_dataset)

    # build dataset per gpu and loader
    subset_dataset = build_subset_per_gpu(opt, val_dataset, log)
    val_loader = DataLoader(subset_dataset,
        batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
    )

    # build runner
    runner = Runner(ckpt_opt, log, save_opt=False)

    # handle use_fp16 for ema
    if opt.use_fp16:
        runner.ema.copy_to() # copy weight from ema to net
        runner.net.diffusion_model.convert_to_fp16()
        runner.net.semantic_enc.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight

    # create save folder
    t_sne_fn = get_t_sne_fn(opt)
    log.info(f"T-SNE plot will be saved to {t_sne_fn}!")

    # feature list create
    images = []
    labels = []
    latent_features = []
    num = 0

    for loader_itr, out in enumerate(val_loader):
        x0, label, _ = compute_batch(ckpt_opt, out)
        x0 = x0.to(opt.device)

        with runner.ema.average_parameters():
            runner.net.semantic_enc.eval()
            z_sem = runner.net.semantic_enc(x0).detach().clone().cpu()

        images.append(x0) # (B, 256, 256)
        labels.append(label) # (B, 1)
        latent_features.append(z_sem) # (B, 512)

        # [-1,1]
        # gathered_latent_features = collect_all_subset(latent_features, log)
        # latent_features.append(gathered_latent_features)

        num += len(z_sem)
        log.info(f"Collected {num} latent featrues!")
        dist.barrier()

    del runner

    images = torch.cat(images, 0)
    # labels = torch.cat(labels, 0).numpy()
    labels = np.concatenate(labels, 0)
    feats = torch.cat(latent_features, axis=0)[:n_samples]

    if opt.global_rank == 0:
        torch.save({"feats": feats}, t_sne_fn / f"feats_{opt.load_itr}.pt")
        log.info(f"Save at {t_sne_fn}")
    dist.barrier()

    log.info(f"Feature extract complete! Collect latent_features={feats.shape}")

    # Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(feats.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    plt.title("Overall Feature Distribution")
    plt.xlabel("Feature Value")
    plt.ylabel("Count")
    plt.savefig(t_sne_fn / f"hists_{opt.load_itr}.png", dpi=300)

    # 2D T-SNE
    # set T-SNE parameter
    n_components = 2
    perplexity = 30
    save_name = f'Ceph_perplexity{perplexity}_seed{opt.seed}_2d'

    # latents = latent_features.reshape(latent_features.shape[0], -1)
    # labels = [i // len(latent_features) for i in range(int(len(latent_features)/3))] \
    #     + [(i // len(latent_features)) + 1 for i in range(int(len(latent_features)/3))] \
    #     + [(i // len(latent_features)) + 2 for i in range(int(len(latent_features)/3))]
    log.info(f"features size: {feats.numpy().shape}")
    log.info(f"labels size: {len(labels)}")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=opt.seed)
    tsne_result = tsne.fit_transform(feats.data)
    log.info(f"T-SNE result shape: {tsne_result.shape}")

    # T-SNE dataframe create
    tsne_df = pd.DataFrame(columns = ['x-tsne', 'y-tsne', 'label'])
    tsne_df['x-tsne'] = tsne_result[:, 0]
    tsne_df['y-tsne'] = tsne_result[:, 1]
    tsne_df['label']  = labels

    # split dataframe according to label
    tsne_df_0 = tsne_df[tsne_df['label'] == 0]
    tsne_df_1 = tsne_df[tsne_df['label'] == 1]
    tsne_df_2 = tsne_df[tsne_df['label'] == 2]

    # 2D scatter plot
    plt.figure(figsize=(6,6))

    plt.scatter(tsne_df_0['x-tsne'], tsne_df_0['y-tsne'], s = 5, color = 'red', label = 'Class1')
    plt.scatter(tsne_df_1['x-tsne'], tsne_df_1['y-tsne'], s = 5, color = 'blue', label = 'Class2')
    plt.scatter(tsne_df_2['x-tsne'], tsne_df_2['y-tsne'], s = 5, color = 'yellow', label = 'Class3')

    plt.xlabel('component 0')
    plt.ylabel('component 1')
    plt.legend()
    plt.savefig(t_sne_fn / f"{save_name}.png", dpi=300)
    log.info('2D T-SNE plot save!')

    # 2D image T-SNE plot
    plot_size = 30000
    max_image_size = 256
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tx = tsne_result[:, 0]
    ty = tsne_result[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    tsne_plot = np.ones(shape=(plot_size, plot_size), dtype=np.uint8) * 255

    plt.figure(figsize=(50,50))

    for image, x, y in tqdm(zip(images, tx, ty), total=len(images)):      
        # draw a rectangle with a color corresponding to the image class
        # image = draw_rectangle_by_class(image, label, palette)

        image = cv2.resize(image.cpu().numpy().squeeze(), (max_image_size, max_image_size), interpolation = cv2.INTER_AREA)
        image = ((image+1)/2*255).astype(np.uint8)
        
        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x] = image

    plt.imshow(tsne_plot, cmap='gray')
    plt.savefig(t_sne_fn / f"{save_name}_image_emb.png", dpi=600)
    plt.close('all')
    log.info('2D T-SNE image embedding plot save!')

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

    # sample
    parser.add_argument("--load-itr",       type=int,  default=160000)
    parser.add_argument("--batch-size",     type=int,  default=10)
    parser.add_argument("--ckpt",           type=str,  default=None,        help="the checkpoint name from which we wish to sample from diffae")
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
