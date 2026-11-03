# -*- coding: utf-8 -*-
"""#199 阶段1 valid 复核 + 预注册融合门槛（纯 CPU / numpy）。

- 10 项格式复核（5849 行、官方顺序、无 NaN、[0,1]、CRLF、无 BOM、SHA256）
- 与参照成员（ens9 / RA_avg / Swin_pair 等）的 Spearman
- 若给了 98.87 的 valid 代理：new = (1-w)*rank(B) + w*rank(member)，只测 w=0.05/0.10，
  报 ΔAUROC、配对 bootstrap P(Δ>0) 与 P(Δ≥+0.0005)、CI95

用法：
  python tools/gate_199_valid.py --csv valid_tta4.csv \
      --proxy proxy_98_87_valid.csv \
      --ref ens9=path --ref ra=path \
      --json out/gate_199.json
不给 --proxy 时只做复核与 Spearman，并在 JSON 里标 gate="MISSING_PROXY"。
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

OFFICIAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "glaucoma_first", "valid.csv")
WEIGHTS = (0.05, 0.10)
THRESHOLD = 0.0005
BOOT = 2000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_pred(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = [r for r in rd if r]
    names = [r[0] for r in rows]
    probs = np.array([float(r[1]) for r in rows], dtype=np.float64)
    return header, names, probs


def rank_norm(v):
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    sv = v[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks / max(n - 1, 1)


def auroc(scores, labels):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    npos = int((y == 1).sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def spearman(a, b):
    return float(np.corrcoef(rank_norm(a), rank_norm(b))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--proxy", default=None, help="98.87 的 valid 代理概率 CSV（5849 行，官方顺序）")
    ap.add_argument("--ref", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--official", default=OFFICIAL)
    ap.add_argument("--json", default=None)
    ap.add_argument("--boot", type=int, default=BOOT)
    args = ap.parse_args()

    header, names, probs = read_pred(args.csv)
    with open(args.official, encoding="utf-8-sig", newline="") as f:
        off = list(csv.DictReader(f))
    off_names = [r["Image_name"] for r in off]
    labels = np.array([int(r["True_Label"]) for r in off], dtype=np.int64)

    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    chk("rows_5849", len(names) == 5849, f"{len(names)} 行")
    chk("header_3cols", header[:2] == ["Image_name", "Probability"], str(header))
    chk("official_order", names == off_names, "与 valid.csv 逐行一致" if names == off_names else "顺序不一致")
    chk("unique_names", len(set(names)) == len(names), f"{len(set(names))} 个唯一名")
    chk("all_finite", bool(np.all(np.isfinite(probs))), "")
    chk("prob_in_0_1", bool(probs.min() >= 0.0 and probs.max() <= 1.0),
        f"min={probs.min():.6f} max={probs.max():.6f}")
    with open(args.csv, "rb") as f:
        raw = f.read()
    chk("crlf", b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n"), "")
    chk("no_bom", not raw.startswith(b"\xef\xbb\xbf"), "")
    if names == off_names:
        chk("label_aligned", True, "无标签列，按官方顺序对齐")
    else:
        chk("label_aligned", False, "顺序不一致，无法对齐标签")

    out = {
        "csv": os.path.abspath(args.csv),
        "sha256": sha256(args.csv),
        "rows": len(names),
        "mean_prob": round(float(probs.mean()), 6),
        "pos_rate": round(float((probs >= 0.5).mean()), 6),
        "checks": checks,
        "auroc_valid": round(auroc(probs, labels), 8) if names == off_names else None,
        "refs": [],
    }

    for spec in args.ref:
        name, path = spec.split("=", 1)
        _, rnames, rprobs = read_pred(path)
        ok = rnames == off_names
        out["refs"].append({
            "name": name, "path": path, "aligned": ok,
            "spearman": round(spearman(probs, rprobs), 6) if ok else None,
        })

    if args.proxy and names == off_names:
        _, pnames, bprobs = read_pred(args.proxy)
        if pnames != off_names:
            out["gate"] = {"status": "PROXY_ORDER_MISMATCH"}
        else:
            base_auc = auroc(bprobs, labels)
            rb = rank_norm(bprobs)
            rm = rank_norm(probs)
            rng = np.random.default_rng(42)
            n = len(labels)
            gate = {"baseline_auroc": round(base_auc, 8), "weights": []}
            for w in WEIGHTS:
                score = (1 - w) * rb + w * rm
                auc = auroc(score, labels)
                deltas = np.empty(args.boot)
                for i in range(args.boot):
                    idx = rng.integers(0, n, n)
                    deltas[i] = auroc(score[idx], labels[idx]) - auroc(rb[idx], labels[idx])
                lo, hi = np.percentile(deltas, [2.5, 97.5])
                gate["weights"].append({
                    "w": w,
                    "auroc": round(auc, 8),
                    "delta": round(auc - base_auc, 8),
                    "p_delta_gt_0": round(float((deltas > 0).mean()), 4),
                    "p_delta_ge_threshold": round(float((deltas >= THRESHOLD).mean()), 4),
                    "ci95": [round(float(lo), 8), round(float(hi), 8)],
                })
            passed = [g for g in gate["weights"] if g["delta"] >= THRESHOLD and g["p_delta_gt_0"] >= 0.95]
            gate["threshold"] = THRESHOLD
            gate["status"] = "PASS" if passed else "NEGATIVE"
            out["gate"] = gate
    else:
        out["gate"] = {"status": "MISSING_PROXY" if not args.proxy else "ORDER_MISMATCH"}

    out["verdict"] = "PASS" if all(c["pass"] for c in checks) else "FAIL"
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
