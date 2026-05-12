from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import random
import argparse

import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.multiprocessing import Process

from logger import Logger
from dataset import dataset
from classifier.runner import Runner

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_training_options():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--name",           type=str,   default=None,         help="experiment ID")
    parser.add_argument("--ckpt",           type=str,   default=None,         help="resumed checkpoint name")
    parser.add_argument("--gpu",            type=str,   default="0",          help="set only if you wish to run on a particular device")
    parser.add_argument("--mode",           type=str,   default="train",      help="train or test or t-sne", choices=["train", "test", "t-sne"])
    parser.add_argument("--transfer-mode",  type=str,   default="finetune",   help="finetune or linear probe", choices=["finetune", "linearprobe"])    
    parser.add_argument("--global-rank",    type=int,   default=0)
    parser.add_argument("--pretrained",     action="store_true",              help="scratch or finetune")

    # --------------- semantic encoder classifier model ---------------
    parser.add_argument("--image-size",     type=int,   default=256)
    parser.add_argument("--num-classes",    type=int,   default=3)
    parser.add_argument("--load-itr",       type=int,   default=80000,        help="checkpoint iteration for loading semantic encoder")
    parser.add_argument("--diffae-ckpt",    type=str,   default=None,         help="the checkpoint name from which we wish to load semantic encoder")
    parser.add_argument("--use-fp16",       action="store_true",              help="use fp16 network weight for faster sampling")
    
    # --------------- optimizer and loss ---------------
    parser.add_argument("--batch-size",     type=int,   default=16)
    parser.add_argument("--num-workers",    type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=5e-3,         help="learning rate")
    parser.add_argument("--lr-gamma",       type=float, default=0.5,          help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=1000,         help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.)
    parser.add_argument("--ema",            type=float, default=0.99)
    parser.add_argument("--resume-epoch",   type=int,   default=0,            help="resumed checkpoint epoch")
    parser.add_argument("--num-epoch",      type=int,   default=200,          help="training epoch")
    parser.add_argument("--save-epoch",     type=int,   default=10,           help="model save epoch")

    # --------------- path and logging ---------------
    parser.add_argument("--dataset-dir",    type=Path,  default="/data",      help="path to dataset")
    parser.add_argument("--print-freq",     type=int,   default=10,           help="print frequency for logging")
    parser.add_argument("--log-dir",        type=Path,  default="log",        help="path to log std outputs and writer data")

    opt = parser.parse_args()

    # ========= auto setup =========
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= opt.gpu
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # opt.device = 'cuda' if opt.gpu is None else f'cuda:{opt.gpu}'

    # ========= path handle =========
    opt.ckpt_path = RESULT_DIR / "classifier" / opt.name
    opt.log_dir = opt.ckpt_path / opt.log_dir
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.ckpt is not None:
        ckpt_file = RESULT_DIR / "classifier" / opt.ckpt / f"{opt.resume_epoch}.pth.tar"
        assert ckpt_file.exists()
        opt.load = ckpt_file
    else:
        opt.load = None

    assert opt.diffae_ckpt is not None

    return opt

def main(opt):
    log = Logger(opt.global_rank, opt.log_dir)
    log.info("=======================================================")
    log.info("            Semantic Encoder Classification            ")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")

    # set seed: make sure each gpu has differnet seed!
    if opt.seed is not None:
        set_seed(opt.seed + opt.global_rank)
    run = Runner(opt, log)

    # build dataset
    if opt.mode == "train":
        train_dataset = dataset.ClassifierDataset(opt, log, mode=opt.mode)
        val_dataset   = dataset.ClassifierDataset(opt, log, mode='valid')
        run.train(opt, log, train_dataset, val_dataset)
    elif opt.mode == "test":
        test_dataset  = dataset.ClassifierDataset(opt, log, mode=opt.mode)
        run.test(opt, log, test_dataset)
    elif opt.mode == "t-sne":
        test_dataset  = dataset.ClassifierDataset(opt, log, mode='test')
        run.t_sne(opt, log, test_dataset)

    log.info("Finish!")

if __name__ == '__main__':
    opt = create_training_options()
    main(opt)