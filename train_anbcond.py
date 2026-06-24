# train_anbcond.py
"""
train.py + ANB(전후방 골격성 각도)를 입력 condition 으로 주입하는 학습 스크립트.
기존 train.py / train_anbloss.py 는 수정하지 않는다.

anbloss(예측) 방식과 달리, ANB 값을 작은 MLP(anb_embed)로 임베딩해 z_sem 에 더하고
(network_anbcond.py), diffusion 복원 loss 만으로 학습한다.
  loss = recon_loss   (ANB 는 보조 loss 가 아니라 입력 condition)
ANB 라벨은 xlsx 에서 읽어 파일 이름(stem)을 File_ID 컬럼과 매칭한다.
anbcond 는 ANB 가 입력이므로 모든 이미지가 매칭되어야 하며, 결측이 있으면
데이터셋 로딩 단계에서 에러로 중단한다.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import random
import argparse

import copy
from pathlib import Path

import numpy as np
import torch
from torch.multiprocessing import Process

from logger import Logger
from distributed_util import init_processes
from dataset.image_dataset_anbcond import PNGGrayDatasetANBCond
from diffae.runner_anbcond import RunnerANBCond

import colored_traceback.always

RESULT_DIR = Path("results")

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def create_training_options():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--name",           type=str,   default=None,        help="experiment ID")
    parser.add_argument("--ckpt",           type=str,   default=None,        help="resumed checkpoint name")
    parser.add_argument("--gpu",            type=int,   default=None,        help="set only if you wish to run on a particular device")
    parser.add_argument("--mode",           type=str,   default="train",     help="train")

    parser.add_argument("--n-gpu-per-node", type=int,   default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,   default='localhost', help="address for master")
    parser.add_argument("--master-port",    type=str,   default='6020',      help="port for master")
    parser.add_argument("--node-rank",      type=int,   default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,   default=1,           help="The number of nodes in multi node env")

    # --------------- DiffAE model ---------------
    parser.add_argument("--image-size",     type=int,   default=256)
    parser.add_argument("--in-channels",    type=int,   default=1)
    parser.add_argument("--t0",             type=float, default=1e-4,        help="sigma start time in network parametrization")
    parser.add_argument("--T",              type=float, default=1.,          help="sigma end time in network parametrization")
    parser.add_argument("--interval",       type=int,   default=1000,        help="number of interval")
    parser.add_argument("--schedule-name",  type=str,   default='linear',    help="target folder name")
    parser.add_argument("--clip-denoise",   action="store_true",             help="clamp predicted image to [-1,1] at each")

    # optional configs for conditional network
    parser.add_argument("--cond-x1",        action="store_true",             help="conditional the network on degraded images")

    # --------------- optimizer and loss ---------------
    parser.add_argument("--batch-size",     type=int,   default=16)
    parser.add_argument("--microbatch",     type=int,   default=1,           help="accumulate gradient over microbatch until full batch-size")
    parser.add_argument("--start-itr",      type=int,   default=0,           help="start or resumed iteration")
    parser.add_argument("--num-itr",        type=int,   default=200000,      help="training iteration")
    parser.add_argument("--lambda-reg",     type=float, default=1.0,         help="weight of regularization loss for semantic encoder")
    parser.add_argument("--lr",             type=float, default=5e-5,        help="learning rate")
    parser.add_argument("--lr-gamma",       type=float, default=0.99,        help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=1000,        help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.0)
    parser.add_argument("--ema",            type=float, default=0.99)

    # --------------- ANB condition ---------------
    parser.add_argument("--anb-xlsx",       type=str,   required=True,       help="ANB 각도가 들어있는 xlsx 경로")
    parser.add_argument("--anb-key-col",    type=str,   default="File_ID",   help="파일 stem 과 매칭할 컬럼명")
    parser.add_argument("--anb-value-col",  type=str,   default="ANB",       help="ANB 각도 값 컬럼명")
    parser.add_argument("--anb-sheet",      type=str,   default="0",         help="시트 이름 또는 인덱스(숫자 문자열이면 인덱스로 사용)")
    parser.add_argument("--anb-hidden",     type=int,   default=256,         help="anb_embed(ANB condition 임베딩) hidden layer 크기")

    # --------------- path and logging ---------------
    parser.add_argument("--dataset-dir",    type=str, nargs='+', default=["/data"], help="paths to dataset(s)")
    parser.add_argument("--image-subdir", type=str, default="png_pre_clahe", help="각 클래스 폴더 안에서 이미지를 읽을 하위 폴더 이름")
    parser.add_argument("--log-dir",        type=Path,  default=".log",      help="path to log std outputs and writer data")
    parser.add_argument("--log-writer",     type=str,   default=None,        help="log writer: can be tensorbard, wandb, or None")
    parser.add_argument("--wandb-api-key",  type=str,   default=None,        help="unique API key of your W&B account; see https://wandb.ai/authorize")
    parser.add_argument("--wandb-user",     type=str,   default=None,        help="user name of your W&B account")
    parser.add_argument("--wandb-project",  type=str,   default=None,        help="W&B project name (default: i2sb)")

    opt = parser.parse_args()

    # 시트 인덱스가 숫자면 int 로 변환(이름이면 문자열 그대로 유지)
    if opt.anb_sheet.isdigit():
        opt.anb_sheet = int(opt.anb_sheet)

    # ========= auto setup =========
    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    assert opt.name is not None
    opt.distributed = opt.n_gpu_per_node > 1
    opt.use_fp16 = False # disable fp16 for training

    # log ngc meta data
    if "NGC_JOB_ID" in os.environ.keys():
        opt.ngc_job_id = os.environ["NGC_JOB_ID"]

    # ========= path handle =========
    os.makedirs(opt.log_dir, exist_ok=True)
    opt.ckpt_path = RESULT_DIR / "diffae" / opt.name
    os.makedirs(opt.ckpt_path, exist_ok=True)

    if opt.ckpt is not None:
        ckpt_file = RESULT_DIR / "diffae" / opt.ckpt / f"{opt.start_itr:0>7}.pt"
        assert ckpt_file.exists()
        opt.load = ckpt_file
    else:
        opt.load = None

    # ========= auto assert =========
    assert opt.batch_size % opt.microbatch == 0, f"{opt.batch_size} is not dividable by {opt.microbatch}!"
    return opt

def main(opt):
    log = Logger(opt.global_rank, opt.log_dir)
    log.info("=======================================================")
    log.info("     Diffusion Autoencoders + ANB conditioning         ")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")

    # set seed: make sure each gpu has differnet seed!
    if opt.seed is not None:
        set_seed(opt.seed + opt.global_rank)

    # build dataset
    train_dataset = PNGGrayDatasetANBCond(opt, log, mode='train')
    # note: images should be normalized to [-1,1] for corruption methods to work properly

    run = RunnerANBCond(opt, log)
    run.train(opt, train_dataset)
    log.info("Finish!")

if __name__ == '__main__':
    opt = create_training_options()

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
            p = Process(target=init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        init_processes(0, opt.n_gpu_per_node, main, opt)
