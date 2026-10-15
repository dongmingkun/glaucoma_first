"""Slim a training checkpoint to a deployment-ready fp16 weight file.

The training `best.pt` contains BOTH the raw model and EMA state_dicts in fp32,
plus validation metadata. For the competition submission package the size limit
is 500 MB (uncompressed), so we:

  1. keep only the inference state_dict (ema/model, whichever had higher val AUROC),
  2. cast weights/bias to fp16 (BatchNorm running stats stay fp32 for accuracy),
  3. save a minimal checkpoint carrying arch/img_size so predict.py can rebuild the model,
  4. print before/after sizes and self-check (rebuild FundusNet + one fp16 forward).

Usage:
  python tools/slim_weights.py --in runs/<exp>/best.pt [--out <path>] [--use auto|ema|model]

Default output name follows the #14 naming convention:
  <arch>_<img>_<cache>_<MMDD>_auprc<0.xxxx>.pt
"""
import argparse
import os
import sys
from datetime import datetime

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "submit_pkg"))
from predict import FundusNet  # noqa: E402  (single source of truth for the model)


def select_use(ckpt, use):
    if use != "auto":
        return use
    auroc_ema = ckpt.get("auroc_ema", -1.0)
    auroc = ckpt.get("auroc", -1.0)
    return "ema" if auroc_ema >= auroc else "model"


def to_fp16_state_dict(sd):
    """Cast float weights to fp16, keep BatchNorm running stats in fp32."""
    out = {}
    for k, v in sd.items():
        if v.is_floating_point() and v.dtype == torch.float32:
            if k.endswith("running_mean") or k.endswith("running_var"):
                out[k] = v
            else:
                out[k] = v.half()
        else:
            out[k] = v
    return out


def human_mb(nbytes):
    return f"{nbytes / 2**20:.1f} MB"


def default_out_name(ckpt, use):
    arch = ckpt.get("arch", "unknown")
    img = ckpt.get("img_size", 0)
    args = ckpt.get("args") or {}
    cr = str(args.get("cache_root", "")) if isinstance(args, dict) else ""
    cache = "768" if "768" in cr else "512"
    auprc = ckpt.get("auprc_ema" if use == "ema" else "auprc", ckpt.get("auprc_ema", ckpt.get("auprc", 0.0)))
    mmdd = datetime.now().strftime("%m%d")
    return f"{arch}_{img}_cache{cache}_{mmdd}_auprc{auprc:.4f}.pt"


def self_check(arch, sd_fp16, img_size):
    """Rebuild FundusNet, load the slim fp16 state_dict, run one fp16 forward."""
    model = FundusNet(arch=arch).half()
    model.load_state_dict(sd_fp16, strict=True)
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, dtype=torch.float16)
    age = torch.tensor([[0.0]], dtype=torch.float16)
    with torch.no_grad():
        logit = model(x, age)
    val = float(logit.item())
    assert val == val, "self-check produced NaN"
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="training best.pt")
    ap.add_argument("--out", default=None, help="output path (default: #14 naming convention)")
    ap.add_argument("--use", default="auto", choices=["auto", "ema", "model"])
    args = ap.parse_args()

    ckpt = torch.load(args.inp, map_location="cpu", weights_only=False)
    use = select_use(ckpt, args.use)
    sd = ckpt.get(use) or ckpt.get("model") or ckpt
    arch = ckpt.get("arch", "efficientnet_b3")
    img_size = ckpt.get("img_size", 384)

    raw_bytes = sum(v.numel() * v.element_size() for v in sd.values())
    sd_fp16 = to_fp16_state_dict(sd)
    slim_bytes = sum(v.numel() * v.element_size() for v in sd_fp16.values())

    out = args.out or os.path.join(os.path.dirname(args.inp), default_out_name(ckpt, use))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    out_ckpt = {
        "state_dict": sd_fp16,          # 通用键（本工具/其他脚本）
        use: sd_fp16,                    # 与 train.py / predict.py 的读取约定对齐（ckpt[use]）
        "arch": arch,
        "img_size": img_size,
        "use": use,
        "fp16": True,
        "source": os.path.basename(args.inp),
    }
    torch.save(out_ckpt, out)

    print(f"源文件: {args.inp}  (use={use}, arch={arch}, img_size={img_size})")
    print(f"瘦身前: {human_mb(raw_bytes)}  ->  瘦身后: {human_mb(slim_bytes)}  "
          f"(节省 {100*(1 - slim_bytes/raw_bytes):.1f}%)")
    print(f"已写出: {out}")

    val = self_check(arch, sd_fp16, img_size)
    print(f"自检通过: FundusNet({arch}).half() 加载 + 一次 fp16 前向 OK (logit={val:.4f})")


if __name__ == "__main__":
    main()
