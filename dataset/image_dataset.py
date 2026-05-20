# dataset/image_dataset.py
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

EXTENSIONS = ('.png', '.jpg', '.jpeg')

CLASS_LIST = ['class1', 'class2', 'class3']
CLASS_DICT = {'class1': 0, 'class2': 1, 'class3': 2}


class PNGGrayDataset(Dataset):
    """
    폴더 구조:
    $DATA_DIR/
        train/
            class1/png/*.png
            class2/png/*.png
            class3/png/*.png
        valid/
        test/
    """
    def __init__(self, opt, log, mode='train'):
        super().__init__()
        self.mode = mode
        self.image_size = opt.image_size
        self.in_channels = getattr(opt, 'in_channels', 1)

        root = Path(opt.dataset_dir) / mode

        # class 서브폴더 → png_post_clahe 서브폴더 탐색
        fnames = []
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            png_dir = class_dir / 'png_post_clahe'
            if png_dir.exists():
                for ext in EXTENSIONS:
                    fnames.extend(sorted(png_dir.rglob(f'*{ext}')))
            else:
                # png_post_clahe 서브폴더 없으면 class 폴더 직접 탐색
                for ext in EXTENSIONS:
                    fnames.extend(sorted(class_dir.rglob(f'*{ext}')))

        self.image_fnames = sorted({str(p) for p in fnames})

        if len(self.image_fnames) == 0:
            raise IOError(f"No PNG/JPG files found under {root}")
        log.info(f"[Dataset] {mode}: {len(self.image_fnames)} images @ {root}")

        # 라벨: 파일 경로에서 class 폴더 이름 추출
        # 경로 예: .../class1/png/IRB....png → class1
        self.labels = []
        for fname in self.image_fnames:
            parts = Path(fname).parts
            label = 0
            for part in parts:
                if part in CLASS_DICT:
                    label = CLASS_DICT[part]
                    break
            self.labels.append(label)

        if mode == 'train':
            self.transform = T.Compose([
                T.Resize((self.image_size, self.image_size), antialias=True),
                T.ToTensor(),
                T.Lambda(lambda t: t * 2 - 1),  # [-1,1]
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
        img = self.transform(img)  # (C, H, W), [-1, 1]
        return img, label, fname