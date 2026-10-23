# -*- coding: utf-8 -*-
"""#109/#111/#118 通用 OOF / test 概率生成（支持自定义 csv + cache 子目录）。

infer.py 只读全局 ``glaucoma_first/{train,valid,test}.csv``，无法对「每折 held-out valid」
做预测，会污染 OOF。本脚本弥补该缺口：对任意 ``--csv``（如 ``folds3/fold{k}_csvroot/valid.csv``）
做 TTA 预测，图从 ``--cache-root/--split-dir`` 下读取。

依赖（与 infer.py/train.py 一致）：``FundusNet`` / ``age_tensor`` 来自 ``train.py``；
``--csv-root`` 在 PR #120 后已由 ``paths.load_rows`` 正确支持，fold 训练不会再静默读全量。

用法：
  # 每折 held-out valid（OOF 用，必须用 last.pt）
  python tools/oof_predict.py --ckpt runs/swin_fold0/last.pt \
      --csv folds3/fold0_csvroot/valid.csv --cache-root cache_all --split-dir valid \
      --img-size 384 --tta 4 --out oof/swin_fold0_valid_tta4.csv

  # test（每折单独出，最后跨折平均）
  python tools/oof_predict.py --ckpt runs/swin_fold0/last.pt \
      --csv glaucoma_first/test.csv --cache-root cache_all --split-dir test \
      --img-size 384 --tta 4 --out oof/swin_fold0_test_tta4.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import FundusNet, age_tensor


class OOFDS(Dataset):
    def __init__(self, rows, img_dir, img_size, flip=0):
        self.rows = rows
        self.img_dir = img_dir
        self.img_size = img_size
        self.flip = flip
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        stem, age = self.rows[i]
        stem = stem.rsplit(".", 1)[0]
        img = Image.open(os.path.join(self.img_dir, stem + ".jpg")).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if self.flip & 1:
            img = torchvision.transforms.functional.hflip(img)
        if self.flip & 2:
            img = torchvision.transforms.functional.vflip(img)
        t = torchvision.transforms.functional.to_tensor(img)
        t = (t - self.mean) / self.std
        return t, age_tensor(age), i


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = np.zeros(len(loader.dataset), dtype=np.float64)
    for x, age, idx in loader:
        x = x.to(device, non_blocking=True)
        age = age.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x, age)
        p = torch.sigmoid(logits.float()).cpu().numpy()
        probs[idx.numpy()] = p
    return probs


def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["Image_name"], float(r["Age"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True, help="要预测的 csv（含 Image_name,Age）")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--split-dir", default="valid", help="cache-root 下的图像子目录")
    ap.add_argument("--img-size", type=int, default=0)
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--use", default="auto", choices=["auto", "ema", "model"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    img_size = args.img_size or ckpt["img_size"]
    use = args.use
    if use == "auto":
        forced = ckpt.get("use")
        if forced in ("ema", "model"):
            use = forced
        else:
            a_ema = ckpt.get("auroc_ema", -1)
            a_mod = ckpt.get("auroc", -1)
            use = "ema" if a_ema >= a_mod else "model"
    print(f"ckpt={args.ckpt} arch={arch} img={img_size} use={use}")

    model = FundusNet(arch, pretrained=False).to(device)
    sd = ckpt["ema"] if use == "ema" else ckpt["model"]
    model.load_state_dict({k: v for k, v in sd.items()}, strict=True)

    img_dir = os.path.join(args.cache_root, args.split_dir)
    rows = load_csv(args.csv)
    flips = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}[args.tta]
    names = [r[0] for r in rows]
    probs = np.zeros(len(rows), dtype=np.float64)
    for fl in flips:
        ds = OOFDS(rows, img_dir, img_size, flip=fl)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, prefetch_factor=4)
        probs += predict(model, dl, device)
    probs /= len(flips)
    print(f"  mean_prob={probs.mean():.5f} n={len(probs)}", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["Image_name", "Probability", "Pred_Label"])
        for n, p in zip(names, probs):
            wr.writerow([n, f"{p:.6f}", 1 if p >= 0.5 else 0])
    print(f"wrote {args.out} ({len(names)} rows)")


if __name__ == "__main__":
    main()
