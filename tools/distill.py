# -*- coding: utf-8 -*-
"""issue #111 — 集成蒸馏训练（云端 GPU 跑）。

把 #111 的教师软标签（tools/gen_soft_labels.py 产物）蒸馏成可部署学生模型。
复用 train.py 的 FundusNet / ModelEMA / evaluate / ThreadLoader / lr_at / FocalBCEWithLogits，
仅把目标从「硬标签」换成「KL + BCE 混合」蒸馏损失，其余训练循环（AMP/EMA/余弦 LR）一致。

损失（二元 KL 蒸馏）：
    L = alpha * L_hard(logits, y)  + (1-alpha) * T^2 * BCE(sigmoid(logits/T), sigmoid(logit_T(y_soft)/T))
其中 L_hard = BCE(pos_weight) 或 Focal（--focal-gamma>0）；教师软标签经温度 T 软化后与学生对齐。

用法：
  # 1) 先生成软标签（教师用 train-only 集成，避免污染 valid）
  python tools/gen_soft_labels.py --teacher runs/a/best.pt --teacher runs/b/best.pt \
      --teacher-weights 0.6 0.4 --split train valid test --tta 4 --out soft_ens.csv
  # 2) 蒸馏学生（默认 train+valid 全量，valid 真标签评估挑 best.pt）
  python tools/distill.py --arch convnext_base --img-size 512 --epochs 10 \
      --soft-labels soft_ens.csv --kd-alpha 0.5 --kd-temp 2.0 \
      --lr 1e-4 --head-lr 5e-4 --pos-weight 5.0 --train-on-all
"""
import argparse
import csv
import math
import os
import random
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from train import (FundusNet, FundusDS, ModelEMA, evaluate, age_tensor,
                   ThreadLoader, set_seed, FocalBCEWithLogits)   # noqa: E402
from paths import load_rows, CACHE_ROOT, CSV_ROOT               # noqa: E402


class KDDS(FundusDS):
    """FundusDS 变体：第 3 路返回 [soft, hard] 拼接（shape (2,)），兼容 ThreadLoader 的 3 元组协议。"""

    def __init__(self, split, rows_soft, img_size, train=True, cache_root=None,
                 rot_deg=25.0, randaugment=None):
        # rows_soft: list of (stem, age, label, soft)
        super().__init__(split, [(s, a, l) for (s, a, l, _) in rows_soft],
                         img_size, train, cache_root, rot_deg, randaugment)
        self.soft = {s: float(sf) for (s, a, l, sf) in rows_soft}

    def __getitem__(self, i):
        t, age, _label = super().__getitem__(i)
        stem = self.rows[i][0]
        soft = torch.tensor(self.soft[stem], dtype=torch.float32)
        return t, age, torch.stack([soft, torch.tensor(self.rows[i][2], dtype=torch.float32)])


def kd_loss(logits, y_hard, y_soft, alpha, T, crit_hard):
    hard = crit_hard(logits.float(), y_hard)
    eps = 1e-4
    yt = y_soft.clamp(eps, 1 - eps)
    t_logit = torch.log(yt / (1 - yt))
    t_prob = torch.sigmoid(t_logit / T)
    s_prob = torch.sigmoid(logits.float() / T)
    soft = F.binary_cross_entropy(s_prob, t_prob, reduction="mean")
    return alpha * hard + (1.0 - alpha) * (T * T) * soft


def lr_at(step, warmup_steps, total_steps, base=1.0):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="convnext_base",
                    help="学生 arch（优先 Base@512/cache768，按显存/延迟选；或 Base@384）")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--pos-weight", type=float, default=5.0)
    ap.add_argument("--focal-gamma", type=float, default=0.0, help=">0 启用 focal 作为硬标签项")
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--init-from", default=None, help="学生骨干热启动（ImageNet 或教师骨干）")
    ap.add_argument("--soft-labels", required=True, help="gen_soft_labels.py 产出的软标签 CSV")
    ap.add_argument("--kd-alpha", type=float, default=0.5, help="硬标签项权重（1-alpha=软标签项）")
    ap.add_argument("--kd-temp", type=float, default=2.0, help="蒸馏温度 T")
    ap.add_argument("--train-on-all", action="store_true",
                    help="在 train+valid 上训练（#111 默认）；不指定则在 train 上训、valid 评估")
    ap.add_argument("--select-metric", default="blend", choices=["auroc", "blend", "auprc"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--csv-root", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    global CACHE_ROOT, CSV_ROOT
    if args.cache_root:
        CACHE_ROOT = args.cache_root
    if args.csv_root:
        CSV_ROOT = args.csv_root

    # 读软标签 -> stem -> soft
    soft = {}
    with open(args.soft_labels, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            stem = r["Image_name"].rsplit(".", 1)[0]
            soft[stem] = float(r["Soft_Label"])
    print(f"软标签加载: {len(soft)} 行")

    set_seed(args.seed)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    torch.backends.cudnn.benchmark = True

    ts = datetime.now().strftime("%m%d_%H%M%S")
    exp = args.out_dir or os.path.join(r"D:\glaucoma_work\runs",
                                       f"distill_{args.arch}_{args.img_size}_{ts}{('_' + args.tag) if args.tag else ''}")
    os.makedirs(exp, exist_ok=True)
    print(f"exp dir: {exp}")

    train_rows = load_rows("train")
    val_rows = load_rows("valid")
    if args.train_on_all:
        base_rows = train_rows + val_rows
    else:
        base_rows = train_rows
    missing = [r[0] for r in base_rows if r[0] not in soft]
    if missing:
        raise SystemExit(f"{len(missing)} 行训练数据缺少软标签（例：{missing[:3]}）；"
                         f"请先用 gen_soft_labels.py 生成覆盖 train+valid 的软标签")
    kd_rows = [(s, a, l, soft[s]) for (s, a, l) in base_rows]
    random.Random(args.seed).shuffle(kd_rows)

    ds_tr = KDDS("train", kd_rows, args.img_size, train=True, cache_root=CACHE_ROOT)
    dl_tr = ThreadLoader(ds_tr, args.batch_size, nthreads=3, shuffle=True, drop_last=True)
    dl_va = None
    if not args.train_on_all:
        ds_va = FundusDS("valid", val_rows, args.img_size, train=False)
        dl_va = ThreadLoader(ds_va, max(32, args.batch_size), nthreads=3, qsize=96,
                             shuffle=False, drop_last=False)

    model = FundusNet(args.arch, pretrained=True).to(device)
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and ("ema" in sd or "model" in sd):
            sd = sd.get(sd.get("use", "ema"), sd.get("ema", sd.get("model")))
        elif isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)
        print(f"warm start from {args.init_from}")

    if args.focal_gamma > 0:
        crit = FocalBCEWithLogits(gamma=args.focal_gamma,
                                  pos_weight=torch.tensor([args.pos_weight], device=device))
    else:
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))

    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        (head_params if (n.startswith("head") or n.startswith("age_mlp")) else backbone_params).append(p)
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    ema = ModelEMA(model, decay=args.ema_decay)

    steps_per_epoch = len(dl_tr) // args.accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_frac))
    base_lrs = [g["lr"] for g in opt.param_groups]

    log_path = os.path.join(exp, "log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "kd_loss", "val_auroc", "val_auprc", "val_auroc_ema", "val_auprc_ema", "lr", "sec"])

    global_step = 0
    best_key = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, nseen, micro = 0.0, 0, 0
        for it, (x, age, y2) in enumerate(dl_tr):
            step = global_step
            fac = lr_at(step, warmup_steps, total_steps)
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * fac
            x = x.to(device, non_blocking=True)
            age = age.to(device, non_blocking=True)
            y_soft = y2[:, 0].to(device, non_blocking=True)
            y_hard = y2[:, 1].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x, age)
                loss = kd_loss(logits.float(), y_hard, y_soft, args.kd_alpha, args.kd_temp, crit) / args.accum
            scaler.scale(loss).backward()
            run_loss += loss.item() * args.accum * len(y_hard)
            nseen += len(y_hard)
            micro += 1
            if micro % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update(model, steps=1)
                global_step += 1
            if (it + 1) % 50 == 0:
                print(f"  ep{epoch} it{it+1}/{len(dl_tr)} kd={run_loss/max(1,nseen):.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e} {time.time()-t0:.0f}s", flush=True)

        train_loss = run_loss / max(1, nseen)
        ckpt = {"model": model.state_dict(), "ema": ema.ema, "arch": args.arch,
                "img_size": args.img_size, "epoch": epoch, "use": "ema",
                "args": vars(args), "distill": True}
        ckpt_path = os.path.join(exp, "last.pt")
        torch.save(ckpt, ckpt_path)
        if args.train_on_all:
            torch.save(ckpt, os.path.join(exp, "best.pt"))
            print(f"epoch {epoch}: kd={train_loss:.4f} -> last.pt+best.pt (train-on-all, no val)", flush=True)
            continue

        auroc, auprc, _, _, _, _ = evaluate(model, dl_va, device, crit)
        backup = ema.apply_to(model)
        auroc_e, auprc_e, _, _, _, _ = evaluate(model, dl_va, device, crit)
        model.load_state_dict(backup)
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch}: kd={train_loss:.4f} | val AUROC={auroc:.4f} AUPRC={auprc:.4f} "
              f"| EMA AUROC={auroc_e:.4f} AUPRC={auprc_e:.4f} ({time.time()-t0:.0f}s)", flush=True)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 5), round(auroc, 5), round(auprc, 5),
                                    round(auroc_e, 5), round(auprc_e, 5), round(lr_now, 8), round(time.time()-t0, 1)])
        key = max(0.5 * (auroc + auprc), 0.5 * (auroc_e + auprc_e))
        if key > best_key:
            best_key = key
            torch.save(ckpt, os.path.join(exp, "best.pt"))
            print(f"  new best -> best.pt ({best_key:.4f})", flush=True)

    print(f"done. best key = {best_key:.4f} -> {exp}/best.pt")


if __name__ == "__main__":
    main()
