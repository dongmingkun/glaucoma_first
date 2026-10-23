# -*- coding: utf-8 -*-
"""issue #111 — 教师软标签生成（蒸馏第一步，云端 GPU 跑）。

加载一个教师模型或一组教师集成，对 train/valid/test 缓存跑推理（TTA1/TTA4），
输出软标签 CSV，供 tools/distill.py 做 KL+BCE 混合蒸馏。

复用 train.py 的 FundusNet / FundusDS，权重 use 字段解析与 train.py 一致。
支持异构集成：每个教师用**自己的** arch/img_size 跑，再按权重平均概率（概率可比）。

用法：
  # 单教师
  python tools/gen_soft_labels.py --teacher runs/x/best.pt --split train valid test --out soft_trainonly.csv
  # 集成（ens9/ens10 的 train-only 部分，按权重平均）
  python tools/gen_soft_labels.py \
      --teacher runs/a/best.pt --teacher runs/b/best.pt \
      --teacher-weights 0.6 0.4 \
      --split train valid test --tta 4 --out soft_ens.csv
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import FundusNet, FundusDS, age_tensor  # noqa: E402
from paths import load_rows, CACHE_ROOT, CSV_ROOT   # noqa: E402


def load_teacher(path, device):
    """返回 (model, arch, img_size)，权重 use 字段解析与 train.py --init-from 一致。"""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    arch = sd.get("arch")
    img_size = sd.get("img_size")
    if isinstance(sd, dict) and ("ema" in sd or "model" in sd or "state_dict" in sd):
        if "use" in sd and sd["use"] in sd:
            state = sd[sd["use"]]
        elif "state_dict" in sd:
            state = sd["state_dict"]
        else:
            a_ema, a_mod = sd.get("auroc_ema", -1), sd.get("auroc", -1)
            state = sd["ema"] if a_ema >= a_mod else sd["model"]
    else:
        state = sd
    model = FundusNet(arch, pretrained=False).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, arch, img_size


@torch.no_grad()
def infer_split(model, split, rows, img_size, device, tta, batch_size, cache_root):
    """对单个 split 跑推理，返回 {stem: prob}（TTA 时平均）。"""
    ds = FundusDS(split, rows, img_size, train=False, cache_root=cache_root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                    pin_memory=True, drop_last=False)
    out = {}
    idx = 0
    for x, age, _ in dl:
        x = x.to(device, non_blocking=True)
        age = age.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(x, age).float())
            if tta >= 2:
                ph = torch.sigmoid(model(torch.flip(x, dims=[3]), age).float())
                pv = torch.sigmoid(model(torch.flip(x, dims=[2]), age).float())
                phv = torch.sigmoid(model(torch.flip(torch.flip(x, dims=[3]), dims=[2]), age).float())
                p = (p + ph + pv + phv) / 4.0
        p = p.cpu().numpy().reshape(-1)
        for j in range(len(p)):
            out[ds.rows[idx + j][0]] = float(p[j])
        idx += len(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", action="append", required=True,
                    help="教师 ckpt 路径，可重复多次（集成）；集成权重用 --teacher-weights")
    ap.add_argument("--teacher-weights", type=float, nargs="*", default=None,
                    help="各教师权重（默认等权）；长度须与 --teacher 一致")
    ap.add_argument("--split", nargs="+", required=True, choices=["train", "valid", "test"])
    ap.add_argument("--tta", type=int, default=1, choices=[1, 4], help="测试时增强（1 或 4）")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=None,
                    help="强制统一 img_size（默认用各教师 ckpt 自带值；异构集成各自用自身值）")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--csv-root", default=None)
    ap.add_argument("--out", required=True, help="输出软标签 CSV（Image_name,Age,Soft_Label[,True_Label]）")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.teacher_weights and len(args.teacher_weights) != len(args.teacher):
        raise SystemExit("--teacher-weights 长度须与 --teacher 一致")
    weights = args.teacher_weights or [1.0] * len(args.teacher)
    wsum = sum(weights)
    weights = [w / wsum for w in weights]

    global CACHE_ROOT, CSV_ROOT
    if args.cache_root:
        CACHE_ROOT = args.cache_root
    if args.csv_root:
        CSV_ROOT = args.csv_root

    device = args.device if torch.cuda.is_available() else "cpu"

    # 逐教师跑（各自 arch/img_size），累加加权概率
    acc = {}          # stem -> 加权概率累加
    cnt = {}          # stem -> 计数（用于等权平均一致性）
    meta = {}         # stem -> (age, true_label)
    for tk, w in zip(args.teacher, weights):
        model, arch, ck_img = load_teacher(tk, device)
        img_size = args.img_size or ck_img
        print(f"[teacher] {tk} arch={arch} img_size={img_size} weight={w:.3f}")
        for split in args.split:
            rows = load_rows(split)
            probs = infer_split(model, split, rows, img_size, device, args.tta,
                                args.batch_size, CACHE_ROOT)
            for (stem, age, label) in rows:
                p = probs[stem]
                acc[stem] = acc.get(stem, 0.0) + w * p
                cnt[stem] = cnt.get(stem, 0) + 1
                meta[stem] = (age, label)
        del model
        torch.cuda.empty_cache()

    # 写出 CSV（按 split 原始顺序合并不便，这里按 stem 排序；distill.py 用 dict 对齐，顺序无关）
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        wcsv = csv.writer(f)
        has_label = any(m[1] >= 0 for m in meta.values())
        wcsv.writerow(["Image_name", "Age", "Soft_Label"] + (["True_Label"] if has_label else []))
        for stem in sorted(acc):
            age, label = meta[stem]
            row = [stem + ".jpg", age, round(acc[stem] / cnt[stem], 6)]
            if has_label:
                row.append(int(label) if label >= 0 else -1)
            wcsv.writerow(row)
    print(f"软标签已写入 {args.out}（{len(acc)} 行，教师数 {len(args.teacher)}，TTA{args.tta}）")


if __name__ == "__main__":
    main()
