"""#193 交付 CSV 的格式 / 合规复核（纯 CPU）。

用法：
  python tools/audit_193_seed7_csv.py --csv oof/cnxl_384_e16_ra_all_s7_test_tta4.csv \
      --ref seed7_s42=path/to/seed7_s42_test_tta4.csv \
      --json out/audit_193_seed7.json

--ref 可重复多次，用来算与既有成员的 Spearman / top200 重合（判断新 seed 是否"又一个同质模型"）。
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np

HEADER = ["Image_name", "Probability", "Pred_Label"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
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
    labels = np.array([int(r[2]) for r in rows], dtype=np.int64)
    return header, names, probs, labels


def rankdata(x):
    """平均秩（与 scipy.stats.rankdata 默认一致），避免额外依赖。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    # 处理并列
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    return (sums / cnt)[inv]


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / math.sqrt((ra * ra).sum() * (rb * rb).sum()))


def top200_overlap(a, b, k=200):
    ta = set(np.argsort(-a, kind="mergesort")[:k])
    tb = set(np.argsort(-b, kind="mergesort")[:k])
    return len(ta & tb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--test-csv", default="glaucoma_first/test.csv")
    ap.add_argument("--ref", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    with open(args.csv, "rb") as f:
        raw = f.read()
    header, names, probs, labels = read_pred(args.csv)

    official = []
    with open(args.test_csv, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            official.append(r["Image_name"])

    chk("rows_5855", len(names) == 5855, f"{len(names)} 行")
    chk("header_3cols", header == HEADER, str(header))
    chk("official_order", names == official,
        f"前 3 个: {names[:3]} vs {official[:3]}" if names[:3] != official[:3] else "与 test.csv 逐行一致")
    chk("unique_names", len(set(names)) == len(names), f"{len(set(names))} 个唯一名")
    chk("all_finite", bool(np.isfinite(probs).all()), "")
    chk("prob_in_0_1", bool(((probs >= 0) & (probs <= 1)).all()),
        f"min={probs.min():.6f} max={probs.max():.6f}")
    chk("label_matches_prob", bool((labels == (probs >= 0.5).astype(np.int64)).all()),
        f"符号不一致 {int((labels != (probs >= 0.5)).sum())} 行")
    chk("crlf", b"\r\n" in raw[:4096] and b"\n" not in raw.replace(b"\r\n", b""), "")
    chk("no_bom", not raw.startswith(b"\xef\xbb\xbf"), "")

    refs = []
    for spec in args.ref:
        name, _, path = spec.partition("=")
        if not os.path.exists(path):
            refs.append({"name": name, "path": path, "status": "MISSING"})
            continue
        _, rnames, rprobs, _ = read_pred(path)
        if rnames == names:
            refs.append({"name": name, "path": path, "status": "ok",
                         "spearman": round(spearman(probs, rprobs), 6),
                         "top200": top200_overlap(probs, rprobs),
                         "max_abs_diff": round(float(np.abs(probs - rprobs).max()), 6)})
        else:
            # 允许按名字对齐
            idx = {n: i for i, n in enumerate(rnames)}
            if set(names) != set(rnames):
                refs.append({"name": name, "path": path, "status": "NAME_MISMATCH"})
                continue
            rp = np.array([rprobs[idx[n]] for n in names])
            refs.append({"name": name, "path": path, "status": "ok(reordered)",
                         "spearman": round(spearman(probs, rp), 6),
                         "top200": top200_overlap(probs, rp),
                         "max_abs_diff": round(float(np.abs(probs - rp).max()), 6)})

    out = {
        "csv": os.path.abspath(args.csv),
        "bytes": len(raw),
        "sha256": sha256(args.csv),
        "md5": hashlib.md5(raw).hexdigest(),
        "mean_prob": round(float(probs.mean()), 6),
        "pos_rate": round(float((probs >= 0.5).mean()), 6),
        "checks": checks,
        "refs": refs,
        "verdict": "PASS" if all(c["pass"] for c in checks) else "FAIL",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
