import os
import numpy as np
import pydicom

from PIL import Image
from pathlib import Path
from functools import partial
from skimage.transform import resize

import torch
import torch.nn as nn
import natsort
from torch.utils.data import Dataset
from torchvision import transforms as T


EXTENSION = ['.jpg', '.jpeg', '.png', '.tiff', '.dcm']

# customize class list if you need
CLASS_LIST = ['class1', 'class2', 'class3']

# customize class dict if you need
CLASS_DICT = {'class1': 0, 'class2': 1, 'class3': 2}

class DiffAEDataset(Dataset):
    def __init__(self, opt, log, mode):
        super().__init__()
        self.mode = mode
        self.dataset_npy = opt.dataset_dir / f"ssj_clean_model_{self.mode}.npy"
        self.image_size = opt.image_size

        # images
        self.fnames = np.load(self.dataset_npy)
        self.image_fnames = natsort.natsorted(fname for fname in self.fnames if self._file_ext(fname) in EXTENSION)
        if len(self.image_fnames) == 0:
            raise IOError('No source image files found in the specified path')
        log.info(f"[Dataset] Built dataset {self.dataset_npy}, size={len(self.image_fnames)}!")

        # labels
        self.labels = [CLASS_DICT[fname.split("/")[-2]] for fname in self.image_fnames]
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1) # [0,1] --> [-1,1]
        ])

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()
    
    def _open_file(self, domain, fname):
        return open(os.path.join(self.dataset_dir, domain, fname), 'rb')
    
    def _crop_image(self, image, point_path):
        point = np.load(point_path)
        min_x = int(np.min(point[:,0]) - (image.shape[1]/20))
        max_x = int(np.max(point[:,0]) + (image.shape[1]/20))
        min_y = int(np.min(point[:,1]) - (image.shape[0]/20))
        max_y = int(np.max(point[:,1]) + (image.shape[0]/20))
        if(min_x < 0):
            min_x = 0
        if(min_y < 0):
            min_y = 0
        if(max_x > image.shape[1]):
            max_x = image.shape[1]
        if(max_y > image.shape[0]):
            max_y = image.shape[0]
        crop_image = image[min_y:max_y, min_x:max_x]

        return crop_image
    
    def _normalize(self, img, mean_std_norm=True):
        if mean_std_norm:
            mean = np.mean(img)
            std = np.std(img)
            img -= mean
            img /= (max(std, 1e-8))
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        return img
    
    def _dcm_preprocess(self, img, point_path, image_size, upper_percentage=99.0, lower_percentage=95):
        img = self._crop_image(img, point_path)

        if img.shape[0] != img.shape[1]:
            if img.shape[0] > img.shape[1]:
                padding = np.zeros(((img.shape[0], (img.shape[0] - img.shape[1]) // 2)), dtype=float)
                padding[:,:] = img.min()
                img = np.concatenate([img, padding, padding], 1)
            elif img.shape[0] < img.shape[1]:
                padding = np.zeros((((img.shape[1] - img.shape[0]) // 2), img.shape[1]), dtype=float)
                padding[:,:] = img.min()
                img = np.concatenate([img, padding, padding], 0)

        upper = np.percentile(img, upper_percentage)
        lower = np.percentile(img, 100-lower_percentage)
        img = np.clip(img, lower, upper)
        img = resize(img, (image_size, image_size))
        img = self._normalize(img)

        return img

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, index):
        fname = self.image_fnames[index]
        label = self.labels[index]

        if self._file_ext(fname) == '.dcm':
            point_path = fname.replace('.dcm', '.npy')
            dcm = pydicom.dcmread(fname, force=True)
            img = dcm.pixel_array.astype(np.float32)
            img = self._dcm_preprocess(img, point_path, self.image_size)

        if img.ndim == 2:
            img = img[:, :, np.newaxis] # HW => HWC
        img = self.transform(img)

        return img, label, fname
    

class ClassifierDataset(Dataset):
    def __init__(self, opt, log, mode):
        super().__init__()
        self.mode = mode
        self.dataset_npy = opt.dataset_dir / f"ssj_clean_classifier_{self.mode}.npy"
        self.image_size = opt.image_size

        # images
        self.fnames = np.load(self.dataset_npy)
        self.image_fnames = natsort.natsorted(fname for fname in self.fnames if self._file_ext(fname) in EXTENSION)
        if len(self.image_fnames) == 0:
            raise IOError('No source image files found in the specified path')
        log.info(f"[Dataset] Built dataset {self.dataset_npy}, size={len(self.image_fnames)}!")

        # labels
        self.labels = [CLASS_DICT[fname.split("/")[-2]] for fname in self.image_fnames]
        
        self.train_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1) # [0,1] --> [-1,1]
        ])

        self.test_transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1) # [0,1] --> [-1,1]
        ])

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()
    
    def _open_file(self, domain, fname):
        return open(os.path.join(self.dataset_dir, domain, fname), 'rb')
    
    def _crop_image(self, image, point_path):
        point = np.load(point_path)
        min_x = int(np.min(point[:,0]) - (image.shape[1]/20))
        max_x = int(np.max(point[:,0]) + (image.shape[1]/20))
        min_y = int(np.min(point[:,1]) - (image.shape[0]/20))
        max_y = int(np.max(point[:,1]) + (image.shape[0]/20))
        if(min_x < 0):
            min_x = 0
        if(min_y < 0):
            min_y = 0
        if(max_x > image.shape[1]):
            max_x = image.shape[1]
        if(max_y > image.shape[0]):
            max_y = image.shape[0]
        crop_image = image[min_y:max_y, min_x:max_x]

        return crop_image
    
    def _normalize(self, img, mean_std_norm=True):
        if mean_std_norm:
            mean = np.mean(img)
            std = np.std(img)
            img -= mean
            img /= (max(std, 1e-8))
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        return (img*255).astype(np.uint8)
    
    def _dcm_preprocess(self, img, point_path, image_size, upper_percentage=99.0, lower_percentage=95):
        img = self._crop_image(img, point_path)

        if img.shape[0] != img.shape[1]:
            if img.shape[0] > img.shape[1]:
                padding = np.zeros(((img.shape[0], (img.shape[0] - img.shape[1]) // 2)), dtype=float)
                padding[:,:] = img.min()
                img = np.concatenate([img, padding, padding], 1)
            elif img.shape[0] < img.shape[1]:
                padding = np.zeros((((img.shape[1] - img.shape[0]) // 2), img.shape[1]), dtype=float)
                padding[:,:] = img.min()
                img = np.concatenate([img, padding, padding], 0)

        upper = np.percentile(img, upper_percentage)
        lower = np.percentile(img, 100-lower_percentage)
        img = np.clip(img, lower, upper)
        img = resize(img, (image_size, image_size))
        img = self._normalize(img)

        return img

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, index):
        fname = self.image_fnames[index]
        label = self.labels[index]

        if self._file_ext(fname) == '.dcm':
            point_path = fname.replace('.dcm', '.npy')
            dcm = pydicom.dcmread(fname, force=True)
            img = dcm.pixel_array.astype(np.float32)
            img = self._dcm_preprocess(img, point_path, self.image_size)

        # if img.ndim == 2:
        #     img = img[:, :, np.newaxis] # HW => HWC

        img = Image.fromarray(img)

        if self.mode == 'train':
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)

        return img, label, fname