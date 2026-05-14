# dataset/image_dataset.py
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

EXTENSIONS = ('.png', '.jpg', '.jpeg')


class PNGGrayDataset(Dataset):
    """
    DiffAE 학습/임베딩 추출용 PNG/JPG grayscale 데이터셋.

    기대 폴더 구조 (클래스 폴더 있어도/없어도 됨):
        $DATA_DIR/
            train/
                a.png, b.png, ...            # flat OK
                or  classA/x.png, classB/y.png   # subfolders OK
            valid/
            test/
    """
    def __init__(self, opt, log, mode='train'):
        super().__init__()
        self.mode = mode
        self.image_size = opt.image_size
        self.in_channels = getattr(opt, 'in_channels', 1)

        root = Path(opt.dataset_dir) / mode
        fnames = []
        for ext in EXTENSIONS:
            fnames.extend(sorted(root.rglob(f'*{ext}')))
            fnames.extend(sorted(root.rglob(f'*{ext.upper()}')))
        self.image_fnames = sorted({str(p) for p in fnames})

        if len(self.image_fnames) == 0:
            raise IOError(f"No PNG/JPG files found under {root}")
        log.info(f"[Dataset] {mode}: {len(self.image_fnames)} images @ {root}")

        # DiffAE 학습은 unconditional → 라벨은 더미 0
        # (하위 폴더 이름을 라벨로 쓰고 싶으면 여기 수정)
        self.labels = [0] * len(self.image_fnames)

        if mode == 'train':
            self.transform = T.Compose([
                T.Resize((self.image_size, self.image_size), antialias=True),
                T.ToTensor(),                       # [0,1]
                T.Lambda(lambda t: t * 2 - 1),      # [-1,1]
            ])
        else:
            self.transform = T.Compose([
                T.Resize((self.image_size, self.image_size), antialias=True),
                T.ToTensor(),
                T.Lambda(lambda t: t * 2 - 1),
            ])

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        fname = self.image_fnames[idx]
        label = self.labels[idx]

        img = Image.open(fname)
        img = img.convert('L' if self.in_channels == 1 else 'RGB')

        img = self.transform(img)           # (C, H, W), [-1, 1]
        return img, label, fname