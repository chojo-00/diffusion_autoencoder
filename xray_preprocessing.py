
"""
xray_preprocessing.py

Lightweight, reproducible preprocessing pipeline for X-ray (e.g., chest radiograph) images.
- DICOM loading with Rescale/LUT & MONOCHROME normalization
- Burned-in annotation & border masking (heuristics)
- Content-aware crop & aspect-preserving resize with padding
- Percentile clipping + normalization ([0,1] or z-score)
- Quality checks (blur score, border fraction, exposure proxy)
- PyTorch Dataset with caching to .npy/.png (16-bit safe)

Dependencies: pydicom, numpy, opencv-python, pillow, torch, tqdm
Optional: pydicom.dataelem for LUT flags, pydicom.pixel_data_handlers
"""

import os
import io
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pydicom
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut

import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# ------------------------------
# Utility helpers
# ------------------------------

def _hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def read_dicom(path: str, prefer_processing: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read DICOM and return (float32 image, metadata dict).
    - Applies RescaleSlope/Intercept
    - Converts MONOCHROME1 -> MONOCHROME2 (i.e., bright = higher value)
    - Optionally applies VOI LUT (only for visualization if prefer_processing is True)
    """
    ds = dcmread(path, force=True, stop_before_pixels=False)

    # Pixel data to array
    try:
        arr = ds.pixel_array.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to read pixel_array: {e} ({path})")

    # Apply Rescale
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    # Photometric: MONOCHROME1 => invert
    pmi = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if pmi.upper() == "MONOCHROME1":
        arr = np.max(arr) - arr

    # VOI LUT: if image is 'for presentation' only; for processing, skip
    if not prefer_processing:
        try:
            arr = apply_voi_lut(arr, ds).astype(np.float32)
        except Exception:
            # VOI LUT may be unavailable, that's okay
            pass

    meta = {
        "PatientID": getattr(ds, "PatientID", ""),
        "StudyInstanceUID": getattr(ds, "StudyInstanceUID", ""),
        "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", ""),
        "SOPInstanceUID": getattr(ds, "SOPInstanceUID", ""),
        "Manufacturer": getattr(ds, "Manufacturer", ""),
        "ModelName": getattr(ds, "ManufacturerModelName", ""),
        "ViewPosition": getattr(ds, "ViewPosition", ""),
        "PatientPosition": getattr(ds, "PatientPosition", ""),
        "BodyPartExamined": getattr(ds, "BodyPartExamined", ""),
        "PixelSpacing": getattr(ds, "PixelSpacing", [np.nan, np.nan]),
        "ImagerPixelSpacing": getattr(ds, "ImagerPixelSpacing", [np.nan, np.nan]),
        "Rows": int(getattr(ds, "Rows", arr.shape[0])),
        "Columns": int(getattr(ds, "Columns", arr.shape[1])),
        "BitsStored": int(getattr(ds, "BitsStored", 16)),
        "PhotometricInterpretation": pmi,
        "BurnedInAnnotation": getattr(ds, "BurnedInAnnotation", ""),
        "Modality": getattr(ds, "Modality", ""),
    }
    return arr.astype(np.float32), meta


# ------------------------------
# Masking burned-in text / borders (heuristics)
# ------------------------------

def detect_borders(img: np.ndarray, thresh: float = 0.98) -> np.ndarray:
    """
    Detect large black borders (collimation) via cumulative projection.
    Returns a boolean mask (True=keep).
    """
    # Normalize to [0,1] for projections
    x = img.copy().astype(np.float32)
    x = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-6)

    h, w = x.shape
    keep = np.ones((h, w), dtype=bool)

    # Row-wise: find leading/trailing runs near zero
    row_mean = x.mean(axis=1)
    col_mean = x.mean(axis=0)

    def find_run(v):
        left = 0
        right = len(v) - 1
        while left < len(v) and v[left] < (1.0 - thresh):
            left += 1
        while right >= 0 and v[right] < (1.0 - thresh):
            right -= 1
        return left, right

    top, bottom = find_run(row_mean)
    left, right = find_run(col_mean)

    keep[:top, :] = False
    keep[bottom+1:, :] = False
    keep[:, :left] = False
    keep[:, right+1:] = False
    return keep


def detect_burned_in_annotations(img: np.ndarray) -> np.ndarray:
    """
    Heuristic text/overlay detection:
    - High-contrast, thin strokes near margins using morphological tophat + Canny.
    Returns a boolean mask (True=keep). Detected overlays are set to False.
    """
    x = img.copy().astype(np.float32)
    x = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-6)
    x8 = (x * 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    tophat = cv2.morphologyEx(x8, cv2.MORPH_TOPHAT, kernel)
    edges = cv2.Canny(tophat, 50, 150)

    # Focus on margins
    h, w = edges.shape
    margin = int(0.12 * min(h, w))
    margin_mask = np.zeros_like(edges, dtype=np.uint8)
    margin_mask[:margin, :] = 1
    margin_mask[-margin:, :] = 1
    margin_mask[:, :margin] = 1
    margin_mask[:, -margin:] = 1

    cand = (edges > 0) & (margin_mask > 0)
    # Slight dilation to cover glyphs
    dil = cv2.dilate(cand.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    keep = np.ones_like(dil, dtype=bool)
    keep[dil > 0] = False
    return keep


def apply_mask(img: np.ndarray, keep_mask: np.ndarray, fill: Optional[float] = None) -> np.ndarray:
    y = img.copy()
    if fill is None:
        fill = float(np.median(img))
    y[~keep_mask] = fill
    return y


# ------------------------------
# Crop & Resize
# ------------------------------

def content_crop(img: np.ndarray, keep_mask: np.ndarray, extra_ratio: float = 0.02) -> np.ndarray:
    ys, xs = np.where(keep_mask)
    if len(ys) == 0 or len(xs) == 0:
        return img  # nothing detected
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    h, w = img.shape
    dy = int((y1 - y0 + 1) * extra_ratio)
    dx = int((x1 - x0 + 1) * extra_ratio)

    y0 = max(0, y0 - dy)
    y1 = min(h - 1, y1 + dy)
    x0 = max(0, x0 - dx)
    x1 = min(w - 1, x1 + dx)

    return img[y0:y1+1, x0:x1+1]


def resize_with_aspect(img: np.ndarray, target: int = 512, pad_value: Optional[float] = None) -> np.ndarray:
    h, w = img.shape
    scale = target / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    # Anti-alias interpolation
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if pad_value is None:
        pad_value = float(np.median(img))

    canvas = np.full((target, target), pad_value, dtype=resized.dtype)
    y0 = (target - nh) // 2
    x0 = (target - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


# ------------------------------
# Intensity normalization
# ------------------------------

def percentile_clip(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi <= lo:
        return x
    x = np.clip(x, lo, hi)
    return x


def normalize(x: np.ndarray, mode: str = "minmax", mask: Optional[np.ndarray] = None) -> np.ndarray:
    if mask is not None:
        vals = x[mask]
    else:
        vals = x

    if mode == "zscore":
        mu = float(np.mean(vals))
        sd = float(np.std(vals) + 1e-6)
        y = (x - mu) / sd
        return y.astype(np.float32)
    else:
        # min-max to [0,1]
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if hi <= lo + 1e-6:
            return np.zeros_like(x, dtype=np.float32)
        y = (x - lo) / (hi - lo)
        return y.astype(np.float32)


# ------------------------------
# Quality metrics
# ------------------------------

def blur_score_var_laplacian(x01: np.ndarray) -> float:
    x8 = (np.clip(x01, 0, 1) * 255).astype(np.uint8)
    return float(cv2.Laplacian(x8, cv2.CV_64F).var())


def border_fraction(keep_mask: np.ndarray) -> float:
    return float(1.0 - keep_mask.mean())


def exposure_proxy(x: np.ndarray) -> float:
    """Median intensity as a crude exposure proxy (post minmax)."""
    x01 = normalize(percentile_clip(x), "minmax")
    return float(np.median(x01))


# ------------------------------
# Configuration
# ------------------------------

@dataclass
class PreprocConfig:
    target_size: int = 512
    clip_low: float = 1.0
    clip_high: float = 99.5
    norm_mode: str = "minmax"  # "minmax" or "zscore"
    prefer_processing: bool = True  # use 'for processing' pipeline; False uses VOI LUT
    mask_borders: bool = True
    mask_burned_in: bool = True
    cache_dir: Optional[str] = None  # if set, cache preprocessed .npy
    use_mask_for_norm: bool = False  # e.g., if you have lung mask; placeholder here
    save_png_16bit: bool = False     # also dump a 16-bit PNG for inspection
    png_dir: Optional[str] = None


def preprocess_once(path: str, cfg: PreprocConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    img, meta = read_dicom(path, prefer_processing=cfg.prefer_processing)

    # Masks
    keep = np.ones_like(img, dtype=bool)
    if cfg.mask_borders:
        keep &= detect_borders(img)
    if cfg.mask_burned_in:
        keep &= detect_burned_in_annotations(img)

    # Apply mask (fill with median) then crop & resize
    img2 = apply_mask(img, keep_mask=keep, fill=None)
    img2 = content_crop(img2, keep_mask=keep)
    img2 = resize_with_aspect(img2, target=cfg.target_size, pad_value=None)

    # Intensity
    img2 = percentile_clip(img2, cfg.clip_low, cfg.clip_high)
    mask_for_norm = None
    if cfg.use_mask_for_norm:
        # Placeholder: if you have a lung mask align it and set here; for now, ignore
        mask_for_norm = None
    img2 = normalize(img2, mode=cfg.norm_mode, mask=mask_for_norm)

    # QA
    qa = {
        "blur_score": blur_score_var_laplacian(img2),
        "border_fraction": border_fraction(keep),
        "exposure_proxy": exposure_proxy(img),
    }

    return img2.astype(np.float32), {"meta": meta, "qa": qa}


# ------------------------------
# Dataset with caching
# ------------------------------

class XRayDataset(Dataset):
    def __init__(self, dicom_paths: List[str], cfg: PreprocConfig):
        self.paths = dicom_paths
        self.cfg = cfg
        if cfg.cache_dir:
            os.makedirs(cfg.cache_dir, exist_ok=True)
        if cfg.save_png_16bit and cfg.png_dir:
            os.makedirs(cfg.png_dir, exist_ok=True)

    def __len__(self):
        return len(self.paths)

    def _cache_key(self, path: str) -> str:
        base = os.path.basename(path)
        h = _hash(path + json.dumps(asdict(self.cfg), sort_keys=True))
        return f"{os.path.splitext(base)[0]}_{h}.npy"

    def _png_key(self, path: str) -> str:
        base = os.path.basename(path)
        h = _hash(path + json.dumps(asdict(self.cfg), sort_keys=True))
        return f"{os.path.splitext(base)[0]}_{h}.png"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]

        # Try cache
        if self.cfg.cache_dir:
            ck = os.path.join(self.cfg.cache_dir, self._cache_key(path))
            meta_path = ck.replace(".npy", ".json")
            if os.path.exists(ck) and os.path.exists(meta_path):
                arr = np.load(ck)
                with open(meta_path, "r") as f:
                    info = json.load(f)
                tensor = torch.from_numpy(arr)[None, ...]  # (1,H,W)
                return {"image": tensor, **info}

        # Compute
        img, info = preprocess_once(path, self.cfg)

        # Save cache
        if self.cfg.cache_dir:
            ck = os.path.join(self.cfg.cache_dir, self._cache_key(path))
            np.save(ck, img.astype(np.float32))
            with open(ck.replace(".npy", ".json"), "w") as f:
                json.dump(info, f, indent=2)

        # Optional: save 16-bit PNG for visual inspection
        if self.cfg.save_png_16bit and self.cfg.png_dir:
            png_path = os.path.join(self.cfg.png_dir, self._png_key(path))
            x01 = np.clip((img - img.min()) / (img.max() - img.min() + 1e-6), 0, 1)
            x16 = (x01 * 65535.0).astype(np.uint16)
            Image.fromarray(x16).save(png_path, format="PNG")

        tensor = torch.from_numpy(img)[None, ...]  # (1,H,W)
        return {"image": tensor, **info}


# ------------------------------
# CLI for batch preprocessing
# ------------------------------

def _scan_dicom(root: str) -> List[str]:
    out = []
    for dp, dn, fn in os.walk(root):
        for f in fn:
            # include typical DICOM extensions and no extension
            if f.lower().endswith((".dcm", ".dicom")) or "." not in f:
                out.append(os.path.join(dp, f))
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser("X-ray preprocessing")
    ap.add_argument("--dicom_dir", type=str, required=True, help="Root folder containing DICOM files")
    ap.add_argument("--cache_dir", type=str, default="./preproc_cache")
    ap.add_argument("--png_dir", type=str, default="./preproc_png")
    ap.add_argument("--target_size", type=int, default=512)
    ap.add_argument("--clip_low", type=float, default=1.0)
    ap.add_argument("--clip_high", type=float, default=99.5)
    ap.add_argument("--norm_mode", type=str, default="minmax", choices=["minmax", "zscore"])
    ap.add_argument("--prefer_processing", action="store_true", help="Prefer 'for processing' (default True)")
    ap.add_argument("--use_voi_lut", action="store_true", help="If set, prefer VOI LUT ('for presentation')")
    ap.add_argument("--no_mask_borders", action="store_true")
    ap.add_argument("--no_mask_burned_in", action="store_true")
    ap.add_argument("--save_png_16bit", action="store_true")

    args = ap.parse_args()

    paths = _scan_dicom(args.dicom_dir)
    print(f"Found {len(paths)} DICOM files under {args.dicom_dir}")

    cfg = PreprocConfig(
        target_size=args.target_size,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        norm_mode=args.norm_mode,
        prefer_processing=not args.use_voi_lut,
        mask_borders=not args.no_mask_borders,
        mask_burned_in=not args.no_mask_burned_in,
        cache_dir=args.cache_dir,
        save_png_16bit=args.save_png_16bit,
        png_dir=args.png_dir,
    )

    os.makedirs(cfg.cache_dir, exist_ok=True)
    if cfg.save_png_16bit:
        os.makedirs(cfg.png_dir, exist_ok=True)

    ds = XRayDataset(paths, cfg)

    blur_scores, border_fracs, exposures = [], [], []
    for i in tqdm(range(len(ds)), desc="Preprocessing"):
        item = ds[i]
        blur_scores.append(item["qa"]["blur_score"])
        border_fracs.append(item["qa"]["border_fraction"])
        exposures.append(item["qa"]["exposure_proxy"])

    # Save a quick dataset-level QA summary
    summary = {
        "n_images": len(ds),
        "blur_score_median": float(np.median(blur_scores)) if len(blur_scores) else None,
        "border_fraction_median": float(np.median(border_fracs)) if len(border_fracs) else None,
        "exposure_proxy_median": float(np.median(exposures)) if len(exposures) else None,
        "config": asdict(cfg),
    }
    with open(os.path.join(cfg.cache_dir, "preproc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Done. Summary saved to", os.path.join(cfg.cache_dir, "preproc_summary.json"))


if __name__ == "__main__":
    main()
