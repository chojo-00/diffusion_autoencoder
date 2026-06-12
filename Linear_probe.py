"""
linear_probe.py — z_sem 선형 프로브 (linear probe)

목적
----
학습이 끝난 DiffAE checkpoint에서 semantic encoder의 z_sem(512-D)을 뽑아,
train split으로 단순 선형 분류기(LogisticRegression)만 학습하고 test split에서
평가한다. "encoder를 freeze한 채 선형 경계만으로 class가 분리되는가"가
z_sem이 class 정보를 linearly decodable하게 담고 있는지에 대한 정량 증거다.

t-sne.py의 추출 루프를 그대로 재사용한다. 추론만 하므로 distributed는 쓰지 않는다.
EMA 가중치로 추출(샘플링과 동일 조건), 정확도를 위해 fp32로 돌린다.

confound 점검(중요)
------------------
z_sem은 skeletal class만이 아니라 촬영기기/kVp/bit-depth 같은 acquisition 요소도
담는다. --meta-csv 를 주면 동일한 프로브를 그 메타 컬럼(예: 검출기 종류)에 대해서도
돌린다. class 프로브와 비교했을 때 메타 쪽도 잘 맞으면 confound를 의심해야 한다.

사용 예
------
    python linear_probe.py \
        --dataset-dir /workspace/.../asan_processing_ver2_..._foldering \
        --image-subdir png_RulerON_CLAHE_ON \
        --ckpt diffae/my_experiment_ver4_Ruler_ON_CLAHE_ON \
        --load-itr 0100000 \
        --image-size 512 \
        --batch-size 16

    # (옵션) 검출기 confound 점검
    #   meta.csv : 컬럼 [fname, device] 형태, fname은 파일명(basename)이면 충분
    python linear_probe.py ... --meta-csv meta.csv --meta-key-col fname --meta-col device
"""

import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)

from logger import Logger
from diffae.runner import Runner
from diffae import ckpt_util
from dataset.image_dataset import PNGGrayDataset, CLASS_LIST

RESULT_DIR = Path("results")


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ #
#  z_sem 추출 (EMA 가중치, fp32)                                      #
# ------------------------------------------------------------------ #
@torch.no_grad()
def extract_z_sem(opt, runner, log, mode):
    out_dir = RESULT_DIR / opt.ckpt / f"probe_iter{opt.load_itr}"
    os.makedirs(out_dir, exist_ok=True)
    feat_fn = out_dir / f"z_sem_{mode}.npy"
    lbl_fn = out_dir / f"labels_{mode}.npy"
    fn_fn = out_dir / f"fnames_{mode}.npy"

    if feat_fn.exists() and lbl_fn.exists() and fn_fn.exists() and not opt.force_extract:
        log.info(f"[Cache] Loading cached features for '{mode}' from {out_dir}")
        return np.load(feat_fn), np.load(lbl_fn), np.load(fn_fn, allow_pickle=True)

    dataset = PNGGrayDataset(opt, log, mode=mode)
    loader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False,
        pin_memory=True, num_workers=opt.num_workers, drop_last=False,
    )

    feats, labels, fnames = [], [], []
    runner.net.eval()
    with runner.ema.average_parameters():           # 샘플링과 동일하게 EMA 사용
        runner.net.semantic_enc.eval()
        for i, (img, label, fname) in enumerate(loader):
            x0 = img.to(torch.float32).to(opt.device)
            z = runner.net.semantic_enc(x0)         # (B, 512)
            feats.append(z.detach().cpu().numpy())
            labels.append(np.asarray(label))
            fnames.extend(list(fname))
            if i % 20 == 0:
                log.info(f"[{mode}] processed ~{i * opt.batch_size} samples")

    feats = np.concatenate(feats, 0)
    labels = np.concatenate(labels, 0)
    fnames = np.asarray(fnames, dtype=object)

    np.save(feat_fn, feats)
    np.save(lbl_fn, labels)
    np.save(fn_fn, fnames)
    log.info(f"[{mode}] z_sem={feats.shape}, labels={labels.shape} -> saved to {out_dir}")
    return feats, labels, fnames


# ------------------------------------------------------------------ #
#  선형 프로브 (표준화 -> LogisticRegression)                         #
# ------------------------------------------------------------------ #
def run_probe(Xtr, ytr, Xte, yte, out_dir, log, tag, label_names):
    n_cls = len(label_names)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    clf = LogisticRegression(
        max_iter=5000, C=1.0, multi_class="multinomial", class_weight="balanced",
    )
    clf.fit(Xtr_s, ytr)

    pred = clf.predict(Xte_s)
    prob = clf.predict_proba(Xte_s)

    acc = accuracy_score(yte, pred)
    bal = balanced_accuracy_score(yte, pred)
    try:
        yte_oh = label_binarize(yte, classes=list(range(n_cls)))
        auc = roc_auc_score(yte_oh, prob, average="macro", multi_class="ovr")
    except Exception as e:           # test에 특정 class가 없으면 AUC 생략
        log.warning(f"[{tag}] AUC skipped: {e}")
        auc = float("nan")

    log.info("=======================================================")
    log.info(f" Linear probe [{tag}]")
    log.info("=======================================================")
    log.info(f"acc={acc:.4f} | balanced_acc={bal:.4f} | macroAUC={auc:.4f} "
             f"(chance acc = {1.0 / n_cls:.3f})")
    log.info("\n" + classification_report(yte, pred, target_names=label_names, digits=3))

    cm = confusion_matrix(yte, pred, labels=list(range(n_cls)))
    ConfusionMatrixDisplay(cm, display_labels=label_names).plot(cmap="Blues", colorbar=False)
    plt.title(f"Linear probe ({tag}) — acc={acc:.3f}, AUC={auc:.3f}")
    plt.savefig(out_dir / f"probe_{tag}_confusion.png", dpi=200, bbox_inches="tight")
    plt.close()

    metrics = {
        "tag": tag, "accuracy": acc, "balanced_accuracy": bal, "macro_auc": auc,
        "chance": 1.0 / n_cls, "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "labels": label_names, "confusion_matrix": cm.tolist(),
    }
    with open(out_dir / f"probe_{tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"[{tag}] saved confusion png + metrics json to {out_dir}")
    return metrics


# ------------------------------------------------------------------ #
#  (옵션) 메타데이터 confound 라벨 매핑                                #
# ------------------------------------------------------------------ #
def map_meta(fnames, csv_path, key_col, val_col, log):
    import pandas as pd
    df = pd.read_csv(csv_path)
    lut = {os.path.basename(str(k)): v for k, v in zip(df[key_col], df[val_col])}
    vals = [lut.get(os.path.basename(str(f)), None) for f in fnames]
    n_hit = sum(v is not None for v in vals)
    log.info(f"[meta] matched {n_hit}/{len(fnames)} files against '{csv_path}'")
    return np.asarray(vals, dtype=object)


def factorize_meta(gtr, gte):
    cats = sorted({v for v in list(gtr) + list(gte) if v is not None}, key=str)
    cat2idx = {c: i for i, c in enumerate(cats)}
    mtr = np.array([v is not None for v in gtr])
    mte = np.array([v is not None for v in gte])
    ytr = np.array([cat2idx[v] for v in gtr[mtr]])
    yte = np.array([cat2idx[v] for v in gte[mte]])
    return ytr, yte, mtr, mte, [str(c) for c in cats]


def main(opt):
    log = Logger(0, ".log")
    set_seed(opt.seed)
    torch.cuda.set_device(0)

    # 학습 때 저장된 options.pkl을 그대로 불러와 네트워크 복원 (use_fp16/device만 덮어씀)
    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)
    runner = Runner(ckpt_opt, log, save_opt=False)

    Xtr, ytr, ftr = extract_z_sem(opt, runner, log, mode="train")
    Xte, yte, fte = extract_z_sem(opt, runner, log, mode="test")
    del runner
    torch.cuda.empty_cache()

    out_dir = RESULT_DIR / opt.ckpt / f"probe_iter{opt.load_itr}"

    # 1) class 프로브 (핵심)
    class_names = CLASS_LIST[:opt.num_classes]
    run_probe(Xtr, ytr, Xte, yte, out_dir, log, tag="class", label_names=class_names)

    # 2) (옵션) confound 프로브
    if opt.meta_csv:
        gtr = map_meta(ftr, opt.meta_csv, opt.meta_key_col, opt.meta_col, log)
        gte = map_meta(fte, opt.meta_csv, opt.meta_key_col, opt.meta_col, log)
        mytr, myte, mtr, mte, meta_names = factorize_meta(gtr, gte)
        if len(meta_names) >= 2 and mtr.sum() > 0 and mte.sum() > 0:
            run_probe(Xtr[mtr], mytr, Xte[mte], myte, out_dir, log,
                      tag=f"meta_{opt.meta_col}", label_names=meta_names)
        else:
            log.warning("[meta] not enough matched samples/categories for a probe.")

    log.info("Finish!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=str, default="0")

    # data (런타임 opt; t-sne.py와 동일하게 PNGGrayDataset이 참조)
    parser.add_argument("--dataset-dir", type=Path, default="/data", help="path to dataset")
    parser.add_argument("--image-subdir", type=str, default="png_pre_clahe",
                        help="각 클래스 폴더 안에서 이미지를 읽을 하위 폴더 이름")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=3)

    # checkpoint
    parser.add_argument("--ckpt", type=str, required=True,
                        help="예: diffae/my_experiment_ver4_Ruler_ON_CLAHE_ON")
    parser.add_argument("--load-itr", type=int, required=True)

    # extraction
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force-extract", action="store_true",
                        help="캐시 무시하고 z_sem 재추출")

    # (옵션) confound 점검
    parser.add_argument("--meta-csv", type=str, default=None,
                        help="fname->그룹 매핑 CSV (예: 검출기/장비)")
    parser.add_argument("--meta-key-col", type=str, default="fname")
    parser.add_argument("--meta-col", type=str, default="device")

    opt = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.use_fp16 = False   # 프로브는 fp32로 (build_ckpt_option이 참조)

    main(opt)