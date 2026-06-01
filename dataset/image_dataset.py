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
        # 클래스 폴더 안에서 이미지를 찾을 하위 폴더 이름 (인자로 주입)
        self.image_subdir = getattr(opt, 'image_subdir', 'png_pre_clahe')

        root = Path(opt.dataset_dir) / mode
        if not root.is_dir():
            raise FileNotFoundError(f"[Dataset] Root folder not found: {root}")

        fnames = []
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            img_dir = class_dir / self.image_subdir
            if not img_dir.is_dir():
                # 폴백 없이 즉시 종료
                raise FileNotFoundError(
                    f"[Dataset] Required subfolder '{self.image_subdir}' not found "
                    f"in class folder: {class_dir}"
                )
            for ext in EXTENSIONS:
                fnames.extend(sorted(img_dir.rglob(f'*{ext}')))

        self.image_fnames = sorted({str(p) for p in fnames})
        if len(self.image_fnames) == 0:
            raise IOError(f"No PNG/JPG files found under {root}/*/{self.image_subdir}")
        log.info(f"[Dataset] {mode}: {len(self.image_fnames)} images @ {root} "
                f"(subdir='{self.image_subdir}')")
        

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