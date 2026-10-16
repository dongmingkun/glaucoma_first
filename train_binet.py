"""BiNet: 双眼中枢模型（双眼联合输入，利用眼间不对称性）

动机：青光眼常为双眼发病但**不对称**（一眼 CDR 明显更大是临床强信号）。
单眼模型看不到对侧眼；本模型把「本眼 + 对侧眼」的骨干特征并联，
并显式给出差分与逐元素乘积，让 head 能直接使用「不对称」这个信号。

数据：利用数据集的成对结构（连续文件 + 同年龄 = 同一患者的双眼）。
- 训练：一对样本产出 2 条监督（以左为目标眼 / 以右为目标眼），两眼的标签都参与损失
- 推理：每张测试图找它的对侧眼（同年龄邻接规则），成对预测；落单的图用自身当对侧

复用 train.py 的网络/EMA/评估工具；本脚本独立，不影响主流水线。
"""
import argparse
import csv
import math
import os
import random
import time
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import ARCH_INFO, FundusNet, AGE_MEDIAN, age_tensor, load_rows, CACHE_ROOT, set_seed, ModelEMA


def build_pairs(rows):
    """连续同年龄 → 同一患者的双眼。返回 [(i, j or -1)]，j=-1 表示落单。"""
    ages = [r[1] for r in rows]
    pairs, i = [], 0
    while i < len(rows):
        if i + 1 < len(rows) and ages[i] == ages[i + 1]:
            pairs.append((i, i + 1))
            i += 2
        else:
            pairs.append((i, -1))
            i += 1
    return pairs


def augment_pair(im1, im2, size):
    """对两只眼施加**完全相同**的几何增广（否则会破坏不对称信号）。"""
    import torchvision
    import torchvision.transforms.functional as F
    # 同参数 RandomResizedCrop：用同一随机框裁剪两张（按各自尺寸归一到同一相对框）
    scale = random.uniform(0.75, 1.0)
    ratio = math.exp(random.uniform(math.log(0.9), math.log(1.1)))
    out = []
    for im in (im1, im2):
        w, h = im.size
        area = w * h * scale
        nw = int(round(math.sqrt(area * ratio)))
        nh = int(round(math.sqrt(area / ratio)))
        nw, nh = min(nw, w), min(nh, h)
        x = random.randint(0, max(0, w - nw))
        y = random.randint(0, max(0, h - nh))
        out.append(F.resize(F.crop(im, y, x, nh, nw), [size, size]))
    if random.random() < 0.5:
        out = [F.hflip(im) for im in out]
    if random.random() < 0.5:
        out = [F.vflip(im) for im in out]
    if random.random() < 0.5:
        ang = random.uniform(-20, 20)
        out = [im.rotate(ang, resample=Image.BILINEAR, fillcolor=(0, 0, 0)) for im in out]
    cj = torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.03)
    out = [cj(im) for im in out]
    return out


class PairDS(Dataset):
    def __init__(self, split, rows, img_size, cache_root, train=True):
        self.dir = os.path.join(cache_root, split)
        self.rows = rows
        self.img_size = img_size
        self.train = train
        self.pairs = build_pairs(rows)
        self.items = []
        for a, b in self.pairs:
            self.items.append((a, b))
            if b >= 0:
                self.items.append((b, a))     # 反向：以另一只眼为目标
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.items)

    def _load(self, i):
        return Image.open(os.path.join(self.dir, self.rows[i][0] + ".jpg")).convert("RGB")

    def __getitem__(self, k):
        i, j = self.items[k]
        im1 = self._load(i)
        im2 = self._load(j if j >= 0 else i)      # 落单：自身当对侧
        if self.train:
            im1, im2 = augment_pair(im1, im2, self.img_size)
        else:
            im1 = im1.resize((self.img_size, self.img_size), Image.BILINEAR)
            im2 = im2.resize((self.img_size, self.img_size), Image.BILINEAR)
        import torchvision
        t1 = (torchvision.transforms.functional.to_tensor(im1) - self.mean) / self.std
        t2 = (torchvision.transforms.functional.to_tensor(im2) - self.mean) / self.std
        return t1, t2, age_tensor(self.rows[i][1]), torch.tensor(self.rows[i][2], dtype=torch.float32)


class BiNet(nn.Module):
    def __init__(self, arch="convnext_small", age_dim=16, dropout=0.3):
        super().__init__()
        base = FundusNet(arch, age_dim=age_dim, pretrained=False, dropout=dropout)
        self.features = base.features
        self.avgpool = base.avgpool
        self.norm = base.norm
        self.feat_dim = base.feat_dim
        self.age_mlp = base.age_mlp
        d = self.feat_dim
        self.head = nn.Sequential(
            nn.Linear(4 * d + age_dim, 512), nn.BatchNorm1d(512), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(512, 1),
        )

    def _feat(self, x):
        f = self.features(x)
        f = self.avgpool(f)
        f = self.norm(f)
        return torch.flatten(f, 1)

    def forward(self, x1, x2, age):
        f1 = self._feat(x1)
        f2 = self._feat(x2)
        a = self.age_mlp(age)
        z = torch.cat([f1, f2, (f1 - f2).abs(), f1 * f2, a], dim=1)
        return self.head(z).squeeze(1)


@torch.no_grad()
def evaluate(model, loader, device, crit):
    model.eval()
    probs, labels = [], []
    for x1, x2, age, y in loader:
        x1, x2, age = x1.to(device), x2.to(device), age.to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x1, x2, age)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(y.numpy())
    p = np.concatenate(probs)
    y = np.concatenate(labels)
    return roc_auc_score(y, p), average_precision_score(y, p), p, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="convnext_small")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--pos-weight", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-from", default=None, help="用单眼模型的 backbone 热启动")
    ap.add_argument("--tag", default="binet")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import paths
    cache_root = args.cache_root or os.environ.get("GLAUCOMA_CACHE_ROOT") or r"D:\glaucoma_work\cache768"
    set_seed(args.seed)
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    ts = datetime.now().strftime("%m%d_%H%M%S")
    exp = os.path.join(r"D:\glaucoma_work\runs", f"binet_{args.arch}_{args.img_size}_{ts}_{args.tag}")
    os.makedirs(exp, exist_ok=True)
    print(f"exp dir: {exp}\ncache_root: {cache_root}", flush=True)

    tr_rows, va_rows = load_rows("train"), load_rows("valid")
    ds_tr = PairDS("train", tr_rows, args.img_size, cache_root, train=True)
    ds_va = PairDS("valid", va_rows, args.img_size, cache_root, train=False)
    nw = args.workers
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=nw,
                       pin_memory=True, drop_last=True, persistent_workers=nw > 0,
                       prefetch_factor=4 if nw > 0 else None)
    dl_va = DataLoader(ds_va, batch_size=max(16, args.batch_size), shuffle=False, num_workers=nw,
                       pin_memory=True, persistent_workers=nw > 0,
                       prefetch_factor=2 if nw > 0 else None)
    print(f"train 对样本 {len(ds_tr)}（含反向） valid {len(ds_va)}", flush=True)

    model = BiNet(args.arch).to(device)
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd = ck.get("ema") if ck.get("auroc_ema", -1) >= ck.get("auroc", -1) else ck.get("model")
        sd = {k: v for k, v in sd.items() if k.startswith(("features.", "avgpool.", "norm."))}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"backbone 热启动: 载入 {len(sd)} 键 (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight], device=device))
    bb, hd = [], []
    for n, p in model.named_parameters():
        (hd if (n.startswith("head") or n.startswith("age_mlp")) else bb).append(p)
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr}, {"params": hd, "lr": args.head_lr}], weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    ema = ModelEMA(model, decay=0.999)
    steps = len(dl_tr) // args.accum * args.epochs
    warm = max(1, int(steps * 0.05))
    base_lrs = [g["lr"] for g in opt.param_groups]

    def lr_at(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, steps - warm)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))

    best = -1.0
    gstep = 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run, nseen, micro = 0.0, 0, 0
        for it, (x1, x2, age, y) in enumerate(dl_tr):
            fac = lr_at(gstep)
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * fac
            x1, x2, age, y = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True), \
                age.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x1, x2, age)
                loss = crit(logits.float(), y) / args.accum
            scaler.scale(loss).backward()
            run += loss.item() * args.accum * len(y)
            nseen += len(y)
            micro += 1
            if micro % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                ema.update(model)
                gstep += 1
            if (it + 1) % 100 == 0:
                print(f"  ep{ep} it{it+1}/{len(dl_tr)} loss={run/max(1,nseen):.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e} vram={torch.cuda.max_memory_allocated()/2**30:.1f}G {time.time()-t0:.0f}s", flush=True)

        auroc, auprc, _, _ = evaluate(model, dl_va, device, crit)
        bk = ema.apply_to(model)
        auroc_e, auprc_e, _, _ = evaluate(model, dl_va, device, crit)
        model.load_state_dict(bk)
        blend = max(0.5 * (auroc + auprc), 0.5 * (auroc_e + auprc_e))
        print(f"epoch {ep}: train={run/max(1,nseen):.4f} AUROC={auroc:.4f} AUPRC={auprc:.4f} | "
              f"EMA AUROC={auroc_e:.4f} AUPRC={auprc_e:.4f} ({time.time()-t0:.0f}s)", flush=True)
        ck = {"model": model.state_dict(), "ema": ema.ema, "arch": "binet_" + args.arch,
              "img_size": args.img_size, "epoch": ep, "auroc": auroc, "auprc": auprc,
              "auroc_ema": auroc_e, "auprc_ema": auprc_e,
              "args": {"cache_root": cache_root, "binet": True}}
        torch.save(ck, os.path.join(exp, "last.pt"))
        key = 0.5 * (auroc_e + auprc_e) if auroc_e >= auroc else 0.5 * (auroc + auprc)
        if key > best:
            best = key
            torch.save(ck, os.path.join(exp, "best.pt"))
            print(f"  new best blend={best:.4f} -> best.pt", flush=True)
    print(f"done. best={best:.4f}")


if __name__ == "__main__":
    main()