# -*- coding: utf-8 -*-
"""同架构多 seed checkpoint 权重平均（model soup），CPU-only。

用途
----
#138(seed7) 与 #145(seed42) 是同一 Base@512 RA train+all 配方的两个 seed。
概率平均（tools/seed_average.py）已经做过；本工具在**权重空间**平均，得到一个新的
单成员 checkpoint，再走正常 test TTA4 / ens10 融合。两者可以互补：

  - 概率平均：两个模型都完整推理，输出空间平均；
  - 权重平均：只推理一个 soup 模型，可能把两个 seed 的噪声一起平滑。

约束
----
- 只支持**同 arch / 同 img_size / 同 state_dict key** 的 checkpoint；
- 同时平均 `ema` 与 `model`（推理默认 `--use ema`）；
- BN 的 `num_batches_tracked` 等整数 buffer 取第一个 checkpoint 的值，不做浮点平均；
- 不评价 valid（train+all 无干净 valid）；是否提分只能由线上提交回答。

用法
----
  python tools/seed_soup.py \
    --ckpts runs/cnxb_512_e16_ra_all/last.pt \
            runs/cnxb_512_e16_ra_all_s42/last.pt \
    --weights 0.5 0.5 \
    --out runs/cnxb_512_e16_ra_all_soup/last.pt

  python tools/seed_soup.py --self-test
"""
import argparse
import os
import sys
import tempfile

import torch


def _load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "ema" not in ckpt:
        sys.exit(f"ERROR: {path} 不是 train.py checkpoint（缺 ema）")
    return ckpt


def _average_state_dicts(states, weights, label):
    keys = set(states[0].keys())
    for i, sd in enumerate(states[1:], 1):
        if set(sd.keys()) != keys:
            sys.exit(f"ERROR: {label} state_dict key 不一致（ckpt #{i}）")
    out = {}
    for k in states[0].keys():
        tensors = [sd[k] for sd in states]
        shapes = {tuple(t.shape) for t in tensors}
        if len(shapes) != 1:
            sys.exit(f"ERROR: {label}.{k} shape 不一致: {shapes}")
        if tensors[0].is_floating_point():
            acc = torch.zeros_like(tensors[0], dtype=torch.float32)
            for w, t in zip(weights, tensors):
                acc.add_(t.float(), alpha=float(w))
            out[k] = acc.to(tensors[0].dtype)
        else:
            # num_batches_tracked 等整数 buffer：取第一个，避免 dtype/语义错误
            out[k] = tensors[0].clone()
    return out


def run(ckpts, weights, out):
    if len(ckpts) < 2:
        sys.exit("ERROR: 至少需要 2 个 checkpoint")
    if weights is None:
        weights = [1.0 / len(ckpts)] * len(ckpts)
    if len(weights) != len(ckpts):
        sys.exit(f"ERROR: --weights 数量 {len(weights)} != ckpt 数 {len(ckpts)}")
    if any(w <= 0 for w in weights):
        sys.exit("ERROR: --weights 必须为正")
    s = float(sum(weights))
    weights = [w / s for w in weights]

    cs = [_load(p) for p in ckpts]
    arch = cs[0].get("arch")
    img_size = cs[0].get("img_size")
    for i, c in enumerate(cs[1:], 1):
        if c.get("arch") != arch:
            sys.exit(f"ERROR: ckpt #{i} arch={c.get('arch')} != {arch}")
        if c.get("img_size") != img_size:
            sys.exit(f"ERROR: ckpt #{i} img_size={c.get('img_size')} != {img_size}")

    soup = dict(cs[0])
    if all("model" in c for c in cs):
        soup["model"] = _average_state_dicts([c["model"] for c in cs], weights, "model")
    soup["ema"] = _average_state_dicts([c["ema"] for c in cs], weights, "ema")
    soup["use"] = "ema"
    soup["soup"] = {
        "sources": [os.path.basename(p) for p in ckpts],
        "weights": [round(float(w), 6) for w in weights],
        "note": "same-arch seed soup; no clean valid for train+all",
    }
    # 避免后续工具把旧单模型的 valid 指标当成 soup 的指标
    soup["auroc"] = -1
    soup["auroc_ema"] = -1
    soup["auprc"] = -1
    soup["auprc_ema"] = -1

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(soup, out)
    print(f"wrote {out}")
    print(f"  arch={arch} img_size={img_size} weights={soup['soup']['weights']}")
    print(f"  sources={soup['soup']['sources']}")
    print("  next: tools/oof_predict.py --ckpt ... --use ema；不要报 valid AUROC")
    return 0


def self_test():
    import csv
    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, "a.pt")
        p2 = os.path.join(td, "b.pt")
        base = {
            "model": {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor(7)},
            "ema": {"w": torch.tensor([2.0, 4.0]), "n": torch.tensor(9)},
            "arch": "convnext_base",
            "img_size": 512,
            "use": "ema",
            "train_on_all": True,
        }
        b = {
            "model": {"w": torch.tensor([5.0, 7.0]), "n": torch.tensor(8)},
            "ema": {"w": torch.tensor([6.0, 8.0]), "n": torch.tensor(10)},
            "arch": "convnext_base",
            "img_size": 512,
            "use": "ema",
            "train_on_all": True,
        }
        torch.save(base, p1)
        torch.save(b, p2)
        out = os.path.join(td, "soup.pt")
        run([p1, p2], [0.5, 0.5], out)
        s = torch.load(out, map_location="cpu", weights_only=False)
        assert torch.allclose(s["model"]["w"], torch.tensor([3.0, 5.0]))
        assert torch.allclose(s["ema"]["w"], torch.tensor([4.0, 6.0]))
        assert s["model"]["n"].item() == 7 and s["ema"]["n"].item() == 9
        assert s["auroc"] == -1 and s["use"] == "ema"
        print("self-test PASS ✔")
    return 0


def main():
    ap = argparse.ArgumentParser(description="same-arch seed checkpoint soup")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--ckpts", nargs="+", default=None)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.ckpts or not args.out:
        sys.exit("需要 --ckpts ... --out ...（或用 --self-test）")
    return run(args.ckpts, args.weights, args.out)


if __name__ == "__main__":
    sys.exit(main())
