import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from pathlib import Path

RESULT_DIR = Path("results/classifier")

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self. count += n
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
class ProgressMeter(object):
    def __init__(self, log, num_batches, meters, prefix=""):
        self.log = log
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.log.info(entries)
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def save_model(name, epoch, network, optimizer): # scheduler
    if isinstance(network, nn.DataParallel):
        torch.save({
            'epoch' : epoch + 1,
            'net': network.module.state_dict(),
            'optimizer' : optimizer.state_dict()
            }, name)
    else:
        torch.save({
            'epoch' : epoch + 1,
            'net': network.state_dict(),
            'optimizer' : optimizer.state_dict()
            }, name)

#===================================================================================================#
#                                       T-SNE Function Handle                                       #
#===================================================================================================#

def compute_batch(out):
    img, label, fpath = out
    x0 = img.detach().to(torch.float32)
    return x0, label, fpath

def get_t_sne_fn(opt):
    t_sne_fn = RESULT_DIR / opt.ckpt / f"t-sne_iter{opt.load_itr}_epoch{opt.resume_epoch}"
    os.makedirs(t_sne_fn, exist_ok=True)
    return t_sne_fn

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