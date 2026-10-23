# -*- coding: utf-8 -*-
"""issue #111 — 教师软标签生成（蒸馏第一步，云端 GPU 跑）。

加载一个教师模型或一组教师集成，对 train/valid/test 缓存跑推理（TTA1/TTA4），
输出软标签 CSV，供 tools/distill.py 做 KL+BCE 混合蒸馏。

两种教师模式：

1) 普通集成（单/多教师）：每个教师对每个 split 跑，按 --teacher-weights 加权平均。
   适合用 train-only 的 ens9/ens10 生成 train/valid/test 软标签。
2) fold-OOF（推荐，配合 #109 的 fold 模型）：
   --fold-valid 给出与 --teacher 一一对应的 held-out fold CSV；
   train/valid 的每一行只取「该行所在 held-out 折」的教师（没见过它），
   test 才用全部教师加权平均。这样 train/valid 软标签无教师泄漏。

OOF 模式的 cache：
  fold CSV 混合了 train_*.jpg 和 valid_*.jpg；`FundusDS` 只在一个 split 目录找图。
  因此需要一个合并 cache：`cache_all/train` 放 train+valid 的全部 57245 张
  （见 folds/README.md 的 `cp -rs` 方案），用 --fold-cache-root 指向它。
  test 仍用原始 cache（--cache-root）。

用法：
  # A. 普通集成（train-only 教师；不要用看过 valid 的 ens10 train+all 家族）
  python tools/gen_soft_labels.py \
      --teacher runs/ens9/best.pt --teacher-weights 1.0 \
      --split train valid test --tta 4 --out soft_trainonly.csv

  # B. fold-OOF（5 折教师；--fold-valid 与 --teacher 顺序一一对应）
  python tools/gen_soft_labels.py \
      --teacher runs/fold0/best.pt --teacher runs/fold1/best.pt \
      --teacher runs/fold2/best.pt --teacher runs/fold3/best.pt --teacher runs/fold4/best.pt \
      --fold-valid folds/fold0_valid.csv folds/fold1_valid.csv \
                   folds/fold2_valid.csv folds/fold3_valid.csv folds/fold4_valid.csv \
      --fold-cache-root cache_all --cache-root cache512 \
      --split train valid test --tta 4 --out soft_fold_oof.csv
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader

from train import FundusNet, FundusDS  # noqa: E402
from paths import load_rows, CACHE_ROOT, AGE_MEDIAN  # noqa: E402


def load_teacher(path, device):
    """返回 (model, arch, img_size)，权重 use 字段解析与 train.py --init-from 一致。"""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    arch = sd.get("arch") if isinstance(sd, dict) else None
    img_size = sd.get("img_size") if isinstance(sd, dict) else None
    if not arch or not img_size:
        raise SystemExit(f"{path} 缺少 arch/img_size 元信息，无法安全推理")
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
def infer_rows(model, rows, dir_split, img_size, device, tta, batch_size, cache_root):
    """对一组行跑推理，返回 {stem: prob}。dir_split 决定在 cache_root 的哪个子目录找图。"""
    ds = FundusDS(dir_split, rows, img_size, train=False, cache_root=cache_root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                    pin_memory=True, drop_last=False)
    out = {}
    idx = 0
    for x, age, _ in dl:
        x = x.to(device, non_blocking=True)
        age = age.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
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


def load_fold_rows(path):
    """读 fold CSV（Image_name,Age,True_Label）-> [(stem, age, label)]。"""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            stem = r["Image_name"].rsplit(".", 1)[0]
            try:
                age = float(r["Age"])
            except (ValueError, TypeError):
                age = AGE_MEDIAN
            if age < 1 or age > 110:
                age = AGE_MEDIAN
            rows.append((stem, min(max(age, 10.0), 100.0), float(r["True_Label"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", action="append", required=True,
                    help="教师 ckpt 路径，可重复多次（集成）；fold 模式下顺序须与 --fold-valid 对应")
    ap.add_argument("--teacher-weights", type=float, nargs="*", default=None,
                    help="各教师权重（默认等权）；长度须与 --teacher 一致；fold 模式只用于 test 集成")
    ap.add_argument("--fold-valid", nargs="*", default=None,
                    help="fold 模式：与 --teacher 一一对应的 held-out fold CSV（train/valid 每行只取对应教师）")
    ap.add_argument("--fold-cache-root", default=None,
                    help="fold 模式：合并 cache（cache_all/train 含 train+valid 全部 57245 张）")
    ap.add_argument("--split", nargs="+", required=True, choices=["train", "valid", "test"])
    ap.add_argument("--tta", type=int, default=1, choices=[1, 4], help="测试时增强（1 或 4）")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=None,
                    help="强制统一 img_size（默认用各教师 ckpt 自带值；异构集成各自用自身值）")
    ap.add_argument("--cache-root", default=None, help="原始 cache（test 用；默认 paths.CACHE_ROOT）")
    ap.add_argument("--csv-root", default=None, help="含 train/valid/test.csv 的目录")
    ap.add_argument("--out", required=True, help="输出软标签 CSV（Image_name,Age,Soft_Label[,True_Label]）")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.teacher_weights and len(args.teacher_weights) != len(args.teacher):
        raise SystemExit("--teacher-weights 长度须与 --teacher 一致")
    weights = args.teacher_weights or [1.0] * len(args.teacher)
    wsum = sum(weights)
    weights = [w / wsum for w in weights]

    if args.fold_valid is not None:
        if len(args.fold_valid) != len(args.teacher):
            raise SystemExit("--fold-valid 数量须与 --teacher 一致（每个 fold 教师对应一个 held-out CSV）")
        if not args.fold_cache_root:
            raise SystemExit("fold 模式必须给 --fold-cache-root（cache_all：train/ 含 train+valid 全部图片）")

    cache_root = args.cache_root or CACHE_ROOT
    device = args.device if torch.cuda.is_available() else "cpu"
    split_rows = {s: load_rows(s, csv_root=args.csv_root) for s in args.split}

    oof_probs = {}      # stem -> 该行 held-out 折教师的概率（fold 模式）
    ens_probs = {}      # (split, stem) -> 加权集成概率（test 或非 fold 模式）

    if args.fold_valid is not None:
        fold_rows = [load_fold_rows(p) for p in args.fold_valid]
        fold_of = {}
        for f, rows in enumerate(fold_rows):
            for stem, _age, _label in rows:
                if stem in fold_of:
                    raise SystemExit(f"{stem} 同时出现在多个 fold valid CSV")
                fold_of[stem] = f
        for s in ("train", "valid"):
            if s not in split_rows:
                continue
            missing = [stem for stem, _a, _l in split_rows[s] if stem not in fold_of]
            if missing:
                raise SystemExit(f"{len(missing)} 行 {s} 不在任何 fold valid CSV（例：{missing[:3]}）")
        # fold CSV 混合 train_*/valid_*.jpg，必须用合并 cache 的 train/ 目录
        missing_imgs = [stem for rows in fold_rows for stem, _a, _l in rows
                        if not os.path.exists(os.path.join(args.fold_cache_root, "train", stem + ".jpg"))]
        if missing_imgs:
            raise SystemExit(
                f"--fold-cache-root 的 train/ 缺 {len(missing_imgs)} 张图（例：{missing_imgs[:3]}）；"
                f"先按 folds/README.md 建 cache_all（train/ 放 train+valid 全部 57245 张）")

        for f, (tk, w) in enumerate(zip(args.teacher, weights)):
            model, arch, ck_img = load_teacher(tk, device)
            img_size = args.img_size or ck_img
            print(f"[teacher {f}] {tk} arch={arch} img_size={img_size} OOF weight=1.0", flush=True)
            probs = infer_rows(model, fold_rows[f], "train", img_size, device, args.tta,
                               args.batch_size, args.fold_cache_root)
            oof_probs.update(probs)
            if "test" in split_rows:
                tp = infer_rows(model, split_rows["test"], "test", img_size, device, args.tta,
                                args.batch_size, cache_root)
                for stem, p in tp.items():
                    ens_probs[("test", stem)] = ens_probs.get(("test", stem), 0.0) + w * p
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    else:
        for f, (tk, w) in enumerate(zip(args.teacher, weights)):
            model, arch, ck_img = load_teacher(tk, device)
            img_size = args.img_size or ck_img
            print(f"[teacher {f}] {tk} arch={arch} img_size={img_size} weight={w:.3f}", flush=True)
            for split in args.split:
                probs = infer_rows(model, split_rows[split], split, img_size, device, args.tta,
                                   args.batch_size, cache_root)
                for stem, p in probs.items():
                    ens_probs[(split, stem)] = ens_probs.get((split, stem), 0.0) + w * p
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # 按 split 原始顺序输出；train/valid 在 fold 模式下用逐行 OOF 教师，test 用集成
    out_rows = []
    for split in args.split:
        for stem, age, label in split_rows[split]:
            if args.fold_valid is not None and split in ("train", "valid"):
                if stem not in oof_probs:
                    raise SystemExit(f"{split} 行 {stem} 没有 OOF 软标签")
                soft = oof_probs[stem]
            else:
                soft = ens_probs[(split, stem)]
            out_rows.append((stem + ".jpg", age, soft, label))

    has_label = any(r[3] >= 0 for r in out_rows)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["Image_name", "Age", "Soft_Label"] + (["True_Label"] if has_label else []))
        for name, age, soft, label in out_rows:
            row = [name, age, round(float(soft), 6)]
            if has_label:
                row.append(int(label) if label >= 0 else -1)
            wcsv.writerow(row)
    print(f"软标签已写入 {args.out}（{len(out_rows)} 行，教师数 {len(args.teacher)}，TTA{args.tta}，"
          f"fold-OOF={'yes' if args.fold_valid is not None else 'no'}）", flush=True)


if __name__ == "__main__":
    main()
