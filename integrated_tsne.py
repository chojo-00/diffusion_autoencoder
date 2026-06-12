import os
import copy
import cv2
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from easydict import EasyDict as edict

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage

from logger import Logger
import distributed_util as dist_util
from diffae.runner import Runner
from dataset.image_dataset import PNGGrayDataset
from diffae import ckpt_util

RESULT_DIR = Path("results")
CLASS_LIST = ['class1', 'class2', 'class3']

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_subset_per_gpu(opt, dataset, log):
    n_data = len(dataset)
    n_gpu  = opt.global_size
    n_dump = (n_data % n_gpu > 0) * (n_gpu - n_data % n_gpu)

    total_idx = np.concatenate([np.arange(n_data), np.zeros(n_dump)]).astype(int)
    idx_per_gpu = total_idx.reshape(-1, n_gpu)[:, opt.global_rank]
    
    indices = idx_per_gpu.tolist()
    subset = Subset(dataset, indices)
    return subset

def build_val_dataset(opt, log):
    val_dataset = PNGGrayDataset(opt, log, mode='test')
    return val_dataset

def get_t_sne_fn(opt):
    t_sne_fn = RESULT_DIR / opt.ckpt / f"t-sne_iter{opt.load_itr}_integrated"
    os.makedirs(t_sne_fn, exist_ok=True)
    return t_sne_fn

def compute_batch(out):
    img, label, fpath = out
    x0 = img.detach().to(torch.float32)
    return x0, label, fpath

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

def png_to_dcm(png_path, image_subdir, dcm_subdir):
    parts = list(Path(png_path).parts)
    idxs = [i for i, part in enumerate(parts) if part == image_subdir]
    if not idxs: return None
    idx = idxs[-1]
    base = Path(*parts[:idx]) if idx > 0 else Path(".")
    rel = Path(*parts[idx + 1:]) if idx + 1 < len(parts) else Path("")
    return (base / dcm_subdir / rel).with_suffix(".dcm")

def read_dicom_meta(dcm_path):
    import pydicom
    try:
        ds = pydicom.dcmread(str(dcm_path), force=True, stop_before_pixels=True)
    except Exception:
        return {}
    def g(tag, default=""):
        v = getattr(ds, tag, default)
        return str(v) if v != "" else default
    
    meta = {
        "manufacturer": g("Manufacturer", "unknown"),
        "model": g("ManufacturerModelName", "unknown"),
        "bits_stored": g("BitsStored", "unknown"),
    }
    try:
        meta["kvp"] = float(getattr(ds, "KVP", np.nan))
    except Exception:
        meta["kvp"] = np.nan
    return meta

def scatter_by_field(coords, series, title, out_path, s=5):
    plt.figure(figsize=(7, 7))
    cats = pd.Series(series).astype(str)
    uniques = sorted(cats.dropna().unique())
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(uniques), 1)))
    for i, u in enumerate(uniques):
        m = (cats == u).values
        plt.scatter(coords[m, 0], coords[m, 1], s=s, color=cmap[i], label=str(u))
    plt.xlabel("Component 0")
    plt.ylabel("Component 1")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def crosstab_report(cluster, df, fields, out_dir, log):
    log.info("\n=== Cross-tabulation: Clusters vs Metadata ===")
    for f in fields:
        if f not in df or df[f].isna().all(): continue
        ct = pd.crosstab(cluster, df[f].astype(str))
        ct_pct = ct.div(ct.sum(axis=1), axis=0).round(3)
        log.info(f"\n[Count] Cluster x {f}:\n{ct}")
        log.info(f"\n[Row %] Cluster x {f}:\n{ct_pct}")
        ct.to_csv(Path(out_dir) / f"crosstab_cluster_x_{f}.csv")

@torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")
    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)
    
    val_dataset = build_val_dataset(opt, log)
    subset_dataset = build_subset_per_gpu(opt, val_dataset, log)
    val_loader = DataLoader(subset_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1)

    runner = Runner(ckpt_opt, log, save_opt=False)
    if opt.use_fp16:
        runner.ema.copy_to()
        runner.net.diffusion_model.convert_to_fp16()
        runner.net.semantic_enc.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99)

    out_dir = get_t_sne_fn(opt)
    log.info(f"Results will be saved to {out_dir}")

    images, labels, latent_features, fpaths = [], [], [], []
    num = 0

    for out in tqdm(val_loader, desc="Extracting Features"):
        x0, label, fpath = compute_batch(out)
        x0 = x0.to(opt.device)

        with runner.ema.average_parameters():
            runner.net.semantic_enc.eval()
            z_sem = runner.net.semantic_enc(x0).detach().cpu()

        images.append(x0.cpu())
        labels.append(label.numpy())
        latent_features.append(z_sem)
        fpaths.extend(list(fpath))

        num += len(z_sem)

    del runner
    images = torch.cat(images, 0)
    labels = np.concatenate(labels, 0).flatten()
    feats = torch.cat(latent_features, 0)

    if opt.global_rank == 0:
        torch.save({"feats": feats}, out_dir / f"feats_{opt.load_itr}.pt")

    # ==========================================
    # 2. Metadata & DICOM Integration
    # ==========================================
    df = pd.DataFrame({
        "path": fpaths,
        "class": [CLASS_LIST[l] for l in labels]
    })

    dcm_subdir = opt.dicom_subdir or opt.image_subdir.replace("png_", "dcm_")
    metas = []
    for p in tqdm(fpaths, desc="Reading DICOM tags"):
        dcm = png_to_dcm(p, opt.image_subdir, dcm_subdir)
        metas.append(read_dicom_meta(dcm) if dcm and dcm.exists() else {})
    
    meta_df = pd.DataFrame(metas)
    if not meta_df.empty:
        for c in meta_df.columns: df[c] = meta_df[c].values
        if "kvp" in df:
            df["kvp_bin"] = pd.cut(df["kvp"], bins=5).astype(str)

    # Merge ANB CSV if provided
    if opt.anb_csv is not None:
        try:
            anb_df = pd.read_csv(opt.anb_csv)
            df['stem'] = df['path'].apply(lambda x: Path(x).stem)
            
            anb_stem_col = 'id' if 'id' in anb_df.columns else anb_df.columns[0]
            anb_df['stem'] = anb_df[anb_stem_col].astype(str).apply(lambda x: Path(x).stem)
            
            df = df.merge(anb_df, on='stem', how='left')
            log.info(f"Successfully merged ANB data. Columns added: {anb_df.columns.tolist()}")
        except Exception as e:
            log.info(f"Failed to merge ANB CSV: {e}")

    # ==========================================
    # 3. T-SNE & KMeans Clustering
    # ==========================================
    log.info("Computing 2D T-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=opt.seed)
    coords = tsne.fit_transform(feats.numpy())
    df['x-tsne'], df['y-tsne'] = coords[:, 0], coords[:, 1]

    # k=2로 큰 덩어리 군집화 파악
    km = KMeans(n_clusters=2, n_init=10, random_state=opt.seed).fit(coords)
    df["cluster"] = km.labels_
    
    # Save visualizations
    scatter_by_field(coords, df["class"], "T-SNE by Class", out_dir / "tsne_class.png")
    scatter_by_field(coords, df["cluster"], "T-SNE by KMeans (k=2)", out_dir / "tsne_cluster.png")
    
    for f in ["manufacturer", "bits_stored", "kvp_bin"]:
        if f in df: scatter_by_field(coords, df[f], f"T-SNE by {f}", out_dir / f"tsne_{f}.png")

    # Image Embedding Plot
    plot_size = 15000
    max_image_size = 128
    tsne_plot = np.ones((plot_size, plot_size), dtype=np.uint8) * 255
    tx, ty = scale_to_01_range(coords[:, 0]), scale_to_01_range(coords[:, 1])

    for img, x, y in zip(images, tx, ty):
        img_np = cv2.resize(img.numpy().squeeze(), (max_image_size, max_image_size))
        img_np = ((img_np + 1) / 2 * 255).astype(np.uint8)
        
        center_x = int((plot_size - max_image_size) * x) + max_image_size // 2
        center_y = int((plot_size - max_image_size) * (1 - y)) + max_image_size // 2
        
        tl_x, tl_y = center_x - max_image_size // 2, center_y - max_image_size // 2
        tsne_plot[tl_y:tl_y+max_image_size, tl_x:tl_x+max_image_size] = img_np

    plt.figure(figsize=(30, 30))
    plt.imshow(tsne_plot, cmap='gray')
    plt.savefig(out_dir / "tsne_image_emb.png", dpi=300)
    
    # Crosstab Report
    fields_to_check = ["class", "manufacturer", "bits_stored", "kvp_bin"]
    crosstab_report(df["cluster"], df, fields_to_check, out_dir, log)

    df.to_csv(out_dir / "latent_metadata_integrated.csv", index=False)
    log.info("Pipeline completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-gpu-per-node", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--num-proc-node", type=int, default=1)

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--dataset-dir", type=Path, default="/data")
    parser.add_argument("--image-subdir", type=str, default="png_pre_clahe")
    parser.add_argument("--dicom-subdir", type=str, default=None)
    parser.add_argument("--anb-csv", type=str, default=None)
    
    parser.add_argument("--load-itr", type=int, default=160000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--use-fp16", action="store_true")

    arg = parser.parse_args()
    opt = edict(distributed=(arg.n_gpu_per_node > 1), device="cuda")
    opt.update(vars(arg))
    
    set_seed(opt.seed)
    
    torch.cuda.set_device(0)
    opt.global_rank, opt.local_rank, opt.global_size = 0, 0, 1
    dist_util.init_processes(0, opt.n_gpu_per_node, main, opt)