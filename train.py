"""Glaucoma binary classification training.

Backbone (ImageNet pretrained) + age MLP fusion head. Handles heavy class
imbalance via pos_weight BCE. AMP fp16, EMA, cosine LR with warmup.

Usage examples:
  python train.py --arch efficientnet_b3 --img-size 384 --epochs 10 --batch-size 32
  python train.py --arch efficientnet_b4 --img-size 448 --epochs 8 --batch-size 24 --accum 2
  python train.py --arch efficientnet_b3 --img-size 384 --epochs 4 --train-on-all --resume-from runs/xxx/best.pt
"""
import argparse
import csv
import math
import os
import random
import sys
import time
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models import (
    EfficientNet_B0_Weights, EfficientNet_B3_Weights, EfficientNet_B4_Weights,
    ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights, ResNet50_Weights,
)

CACHE_ROOT = r"D:\glaucoma_work\cache512"
CSV_ROOT = r"C:\Users\Kevin\Desktop\glaucoma_first\glaucoma_first"
AGE_MEDIAN = 58.0

ARCH_INFO = {
    "efficientnet_b0": (torchvision.models.efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, 1280),
    "efficientnet_b3": (torchvision.models.efficientnet_b3, EfficientNet_B3_Weights.DEFAULT, 1536),
    "efficientnet_b4": (torchvision.models.efficientnet_b4, EfficientNet_B4_Weights.DEFAULT, 1792),
    "convnext_tiny": (torchvision.models.convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT, 768),
    "convnext_small": (torchvision.models.convnext_small, ConvNeXt_Small_Weights.DEFAULT, 768),
    "convnext_base": (torchvision.models.convnext_base, ConvNeXt_Base_Weights.DEFAULT, 1024),
    "resnet50": (torchvision.models.resnet50, ResNet50_Weights.DEFAULT, 2048),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FundusNet(nn.Module):
    def __init__(self, arch="efficientnet_b3", age_dim=16, pretrained=True, dropout=0.3):
        super().__init__()
        ctor, weights, feat_dim = ARCH_INFO[arch]
        self.arch = arch
        m = ctor(weights=weights if pretrained else None)
        if arch.startswith("efficientnet"):
            self.features = m.features
            self.avgpool = m.avgpool
            self.norm = nn.Identity()
        elif arch.startswith("convnext"):
            self.features = m.features
            self.avgpool = m.avgpool
            self.norm = m.classifier[0]  # LayerNorm2d
        elif arch.startswith("resnet"):
            self.features = nn.Sequential(*list(m.children())[:-2])
            self.avgpool = m.avgpool
            self.norm = nn.Identity()
        else:
            raise ValueError(arch)
        self.feat_dim = feat_dim
        self.age_mlp = nn.Sequential(
            nn.Linear(1, age_dim), nn.SiLU(), nn.Linear(age_dim, age_dim), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim + age_dim, 512), nn.BatchNorm1d(512), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(512, 1),
        )

    def forward(self, x, age):
        f = self.features(x)
        f = self.avgpool(f)
        f = self.norm(f)
        f = torch.flatten(f, 1)
        a = self.age_mlp(age)
        return self.head(torch.cat([f, a], dim=1)).squeeze(1)


def load_rows(split):
    path = os.path.join(CSV_ROOT, split + ".csv")
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            name = r["Image_name"]
            stem = name.rsplit(".", 1)[0]
            try:
                age = float(r["Age"])
            except (ValueError, TypeError):
                age = AGE_MEDIAN
            if age < 1 or age > 110:
                age = AGE_MEDIAN
            label = float(r["True_Label"]) if "True_Label" in r else -1.0
            rows.append((stem, min(max(age, 10.0), 100.0), label))
    return rows


def age_tensor(age):
    return torch.tensor([(age - AGE_MEDIAN) / 15.0], dtype=torch.float32)


class FundusDS(Dataset):
    def __init__(self, split, rows, img_size=384, train=True):
        self.dir = os.path.join(CACHE_ROOT, split)
        self.rows = rows
        self.img_size = img_size
        self.train = train
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        stem, age, label = self.rows[i]
        img = Image.open(os.path.join(self.dir, stem + ".jpg")).convert("RGB")
        if self.train:
            s = self.img_size
            # random resized crop, fundus-friendly scale
            rrc = torchvision.transforms.RandomResizedCrop(
                s, scale=(0.7, 1.0), ratio=(0.85, 1.18), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            img = rrc(img)
            if random.random() < 0.5:
                img = torchvision.transforms.functional.hflip(img)
            if random.random() < 0.5:
                img = torchvision.transforms.functional.vflip(img)
            if random.random() < 0.5:
                img = img.rotate(random.uniform(-25, 25), resample=Image.BILINEAR, fillcolor=(0, 0, 0))
            cj = torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04)
            img = cj(img)
            if random.random() < 0.15:
                img = img.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(random.uniform(0.3, 1.2)))
        else:
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        t = torchvision.transforms.functional.to_tensor(img)
        t = (t - self.mean) / self.std
        if self.train and random.random() < 0.1:
            # random erasing
            h, w = t.shape[1:]
            eh, ew = int(h * random.uniform(0.05, 0.2)), int(w * random.uniform(0.05, 0.2))
            y, x = random.randint(0, h - eh), random.randint(0, w - ew)
            t[:, y:y + eh, x:x + ew] = torch.randn(3, eh, ew)
        return t, age_tensor(age), torch.tensor(label, dtype=torch.float32)


def select_score(auroc, auprc, mode):
    if mode == "auroc":
        return auroc
    if mode == "auprc":
        return auprc
    return 0.5 * (auroc + auprc)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
        self.decay = decay
        self.n = 0

    @torch.no_grad()
    def update(self, model, steps=1):
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        d = d ** steps
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.ema[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                self.ema[k] = v.detach().clone()
        self.n += steps

    def apply_to(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v for k, v in self.ema.items()}, strict=True)
        return backup


class ThreadLoader:
    """In-process threaded prefetcher.

    Windows multiprocessing DataLoader workers cost ~1.25 GB commit each and the
    system commit limit (RAM+pagefile) is the binding constraint on this machine.
    PIL/numpy release the GIL, so a handful of threads decode+augment while the
    main thread runs the GPU step, with no extra process memory.
    """

    def __init__(self, ds, batch_size, nthreads=3, qsize=48, shuffle=True, drop_last=True):
        self.ds = ds
        self.bs = batch_size
        self.nthreads = nthreads
        self.qsize = qsize
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        n = len(self.ds) // self.bs
        return n if self.drop_last else math.ceil(len(self.ds) / self.bs)

    def __iter__(self):
        import queue as _q
        import threading

        order = list(range(len(self.ds)))
        if self.shuffle:
            random.shuffle(order)
        idx_q = _q.Queue()
        for i in order:
            idx_q.put(i)
        out_q = _q.Queue(maxsize=self.qsize)

        def work():
            while True:
                try:
                    i = idx_q.get_nowait()
                except _q.Empty:
                    return
                out_q.put(self.ds[i])

        threads = [threading.Thread(target=work, daemon=True) for _ in range(self.nthreads)]
        for t in threads:
            t.start()
        batch = []
        done = 0
        while True:
            try:
                item = out_q.get(timeout=2.0)
            except _q.Empty:
                if not any(t.is_alive() for t in threads):
                    break
                continue
            batch.append(item)
            if len(batch) == self.bs:
                xs = torch.stack([b[0] for b in batch])
                ages = torch.stack([b[1] for b in batch])
                ys = torch.stack([b[2] for b in batch])
                yield xs, ages, ys
                done += len(batch)
                batch = []
        if batch and not self.drop_last:
            xs = torch.stack([b[0] for b in batch])
            ages = torch.stack([b[1] for b in batch])
            ys = torch.stack([b[2] for b in batch])
            yield xs, ages, ys


def proc_commit_mb():
    import ctypes
    import ctypes.wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_void_p, ctypes.wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    p = PMC()
    p.cb = ctypes.sizeof(PMC)
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(p), p.cb):
        return -1.0
    return p.PrivateUsage / 2**20


@torch.no_grad()
def evaluate(model, loader, device, crit):
    model.eval()
    probs, labels, losses = [], [], []
    for x, age, y in loader:
        x, age, y = x.to(device, non_blocking=True), age.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x, age)
            loss = crit(logits.float(), y)
        probs.append(torch.sigmoid(logits.float()).cpu())
        labels.append(y.cpu())
        losses.append(loss.item() * len(y))
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    # best balanced accuracy over all thresholds (relevant if the LB metric is threshold-based)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, probs)
    balacc_max = float(np.max(0.5 * (tpr + (1 - fpr))))
    return auroc, auprc, sum(losses) / len(labels), probs, labels, balacc_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b3", choices=list(ARCH_INFO))
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--pos-weight", type=float, default=5.0)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--loader", default="threads", choices=["threads", "workers"])
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=int, default=0, help="train on random subset (smoke test)")
    ap.add_argument("--max-steps-per-epoch", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--train-on-all", action="store_true", help="add valid into train, no val split (final model)")
    ap.add_argument("--init-from", default=None, help="load weights only (state_dict) to warm start")
    ap.add_argument("--select-metric", default="blend", choices=["auroc", "blend", "auprc"],
                    help="checkpoint 选择标准: auroc | blend=0.5*(auroc+auprc) | auprc")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    ts = datetime.now().strftime("%m%d_%H%M%S")
    exp = args.out_dir or os.path.join(r"D:\glaucoma_work\runs",
                                       f"{args.arch}_{args.img_size}_{ts}{('_' + args.tag) if args.tag else ''}")
    os.makedirs(exp, exist_ok=True)
    print(f"exp dir: {exp}")

    train_rows = load_rows("train")
    val_rows = load_rows("valid")
    if args.train_on_all:
        all_rows = train_rows + val_rows
        random.Random(args.seed).shuffle(all_rows)
        train_rows = all_rows
    if args.subset:
        rng = random.Random(args.seed)
        pos = [r for r in train_rows if r[2] == 1]
        neg = [r for r in train_rows if r[2] == 0]
        k = args.subset
        train_rows = rng.sample(pos, min(len(pos), max(1, int(k * 0.032)))) + rng.sample(neg, min(len(neg), k))
        random.Random(args.seed).shuffle(train_rows)
        print(f"subset train rows: {len(train_rows)}")
    npos = sum(1 for r in train_rows if r[2] == 1)
    print(f"train rows={len(train_rows)} pos={npos} ({npos/max(1,len(train_rows)):.4f})")

    ds_tr = MixedDS(train_rows, args.img_size) if args.train_on_all else \
        FundusDS("train", train_rows, args.img_size, train=True)
    ds_va = FundusDS("valid", val_rows, args.img_size, train=False)
    if args.loader == "threads":
        dl_tr = ThreadLoader(ds_tr, args.batch_size, nthreads=args.threads, shuffle=True, drop_last=True)
        dl_va = ThreadLoader(ds_va, max(32, args.batch_size), nthreads=args.threads, qsize=96,
                             shuffle=False, drop_last=False)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                           pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
                           prefetch_factor=4 if args.workers > 0 else None)
        dl_va = DataLoader(ds_va, batch_size=max(32, args.batch_size), shuffle=False,
                           num_workers=max(2, args.workers // 2), pin_memory=True,
                           persistent_workers=False, prefetch_factor=2)

    model = FundusNet(args.arch, pretrained=True).to(device)
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and ("ema" in sd or "model" in sd):
            a_ema, a_mod = sd.get("auroc_ema", -1), sd.get("auroc", -1)
            sd = sd["ema"] if a_ema >= a_mod else sd["model"]
        model.load_state_dict(sd, strict=True)
        print(f"warm start from {args.init_from}")

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
    if args.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_frac))
    base_lrs = [g["lr"] for g in opt.param_groups]

    def lr_at(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))

    log_path = os.path.join(exp, "log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "auroc", "auprc", "val_auroc_ema", "val_auprc_ema", "lr", "sec"])

    global_step = 0
    best_auroc = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, nseen, micro = 0.0, 0, 0
        for it, (x, age, y) in enumerate(dl_tr):
            if args.max_steps_per_epoch and it >= args.max_steps_per_epoch * args.accum:
                break
            step = global_step
            fac = lr_at(step)
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * fac
            x = x.to(device, non_blocking=True)
            age = age.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x, age)
                loss = crit(logits.float(), y) / args.accum
            scaler.scale(loss).backward()
            run_loss += loss.item() * args.accum * len(y)
            nseen += len(y)
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
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"  ep{epoch} it{it+1}/{len(dl_tr)} loss={run_loss/max(1,nseen):.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e} vram={mem:.1f}G {time.time()-t0:.0f}s", flush=True)

        train_loss = run_loss / max(1, nseen)
        if args.train_on_all:
            torch.save({"model": model.state_dict(), "ema": ema.ema, "arch": args.arch,
                        "img_size": args.img_size, "epoch": epoch}, os.path.join(exp, "last.pt"))
            print(f"epoch {epoch}: train_loss={train_loss:.4f} ({time.time()-t0:.0f}s) saved last.pt")
            continue

        auroc, auprc, vloss, _, _, balmax = evaluate(model, dl_va, device, crit)
        backup = ema.apply_to(model)
        auroc_e, auprc_e, vloss_e, _, _, balmax_e = evaluate(model, dl_va, device, crit)
        model.load_state_dict(backup)
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {epoch}: train={train_loss:.4f} val={vloss:.4f} AUROC={auroc:.4f} AUPRC={auprc:.4f} "
              f"balmax={balmax:.4f} | EMA AUROC={auroc_e:.4f} AUPRC={auprc_e:.4f} balmax={balmax_e:.4f} "
              f"({time.time()-t0:.0f}s) commit={proc_commit_mb():.0f}MB", flush=True)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 5), round(vloss, 5), round(auroc, 5), round(auprc, 5),
                                    round(auroc_e, 5), round(auprc_e, 5), round(lr_now, 8), round(time.time() - t0, 1)])
        ckpt = {"model": model.state_dict(), "ema": ema.ema, "arch": args.arch, "img_size": args.img_size,
                "epoch": epoch, "auroc": auroc, "auroc_ema": auroc_e, "auprc": auprc, "auprc_ema": auprc_e,
                "val_rows": [(r[0], r[1]) for r in val_rows],
                "args": vars(args)}
        torch.save(ckpt, os.path.join(exp, "last.pt"))
        key = max(select_score(auroc, auprc, args.select_metric),
                  select_score(auroc_e, auprc_e, args.select_metric))
        if key > best_auroc:
            best_auroc = key
            torch.save(ckpt, os.path.join(exp, "best.pt"))
            print(f"  new best ({args.select_metric}): {best_auroc:.4f} -> best.pt", flush=True)

    print(f"done. best AUROC = {best_auroc:.4f}")


class MixedDS(Dataset):
    """Train-on-all: rows from train+valid, resolved by existing cache path."""

    def __init__(self, rows, img_size=384):
        self.rows = rows
        self.img_size = img_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        stem, age, label = self.rows[i]
        p1 = os.path.join(CACHE_ROOT, "train", stem + ".jpg")
        p2 = os.path.join(CACHE_ROOT, "valid", stem + ".jpg")
        path = p1 if os.path.exists(p1) else p2
        img = Image.open(path).convert("RGB")
        s = self.img_size
        rrc = torchvision.transforms.RandomResizedCrop(
            s, scale=(0.7, 1.0), ratio=(0.85, 1.18), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        img = rrc(img)
        if random.random() < 0.5:
            img = torchvision.transforms.functional.hflip(img)
        if random.random() < 0.5:
            img = torchvision.transforms.functional.vflip(img)
        if random.random() < 0.5:
            img = img.rotate(random.uniform(-25, 25), resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        cj = torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04)
        img = cj(img)
        if random.random() < 0.15:
            from PIL import ImageFilter
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
        t = torchvision.transforms.functional.to_tensor(img)
        t = (t - self.mean) / self.std
        if random.random() < 0.1:
            h, w = t.shape[1:]
            eh, ew = int(h * random.uniform(0.05, 0.2)), int(w * random.uniform(0.05, 0.2))
            y, x = random.randint(0, h - eh), random.randint(0, w - ew)
            t[:, y:y + eh, x:x + ew] = torch.randn(3, eh, ew)
        return t, age_tensor(age), torch.tensor(label, dtype=torch.float32)


if __name__ == "__main__":
    main()