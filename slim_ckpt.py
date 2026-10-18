"""fp16 瘦身：把 train.py 产出的 fp32 best.pt 转为提交包用的 fp16 精简权重。

对齐 cs_hi512_v2.pt 的瘦身格式：{state_dict, ema, arch, img_size, use, fp16, source}
"""
import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="fp32 best.pt")
    ap.add_argument("--out", required=True, help="fp16 瘦身输出 .pt")
    args = ap.parse_args()

    c = torch.load(args.inp, map_location="cpu", weights_only=False)
    use = c.get("use", "ema")
    if use not in ("ema", "model"):
        use = "ema" if c.get("auroc_ema", -1) >= c.get("auroc", -1) else "model"

    model_key = "model" if "model" in c else "state_dict"
    slim = {
        "state_dict": {k: v.half() for k, v in c[model_key].items()},
        "ema": {k: v.half() for k, v in c["ema"].items()},
        "arch": c["arch"],
        "img_size": c["img_size"],
        "use": use,
        "fp16": True,
        "train_on_all": c.get("train_on_all", False),
        "source": os.path.basename(args.inp),
    }
    torch.save(slim, args.out)
    print(f"slim: {args.inp} -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) "
          f"arch={c['arch']} img={c['img_size']} use={use} train_on_all={slim['train_on_all']}")


if __name__ == "__main__":
    main()
