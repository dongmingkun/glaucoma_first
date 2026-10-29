# -*- coding: utf-8 -*-
"""#185 的 CPU 侧核验：seed123 成员是否合规 + check_train_on_all 的可用替代指标。

`check_train_on_all.py` 需要 ref checkpoint + cache_all768（GPU/数十 GB），
本机没有，因此这里用**纯 CSV 口径**给出等价判据：
成员校验（行数/名序/NaN/范围）、与 seed7/seed42 的 Spearman 与 top200、
mean 漂移，以及把 seedavg 换成 3-seed 后 98.87 配方的位移。

用法：python tools/check_seed123_cpu.py [--out docs/audit_185/check_seed123.json]
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S7 = "cnxb_512_e16_ra_all_test_tta4.csv"
S42 = "cnxb_512_e16_ra_all_s42_test_tta4.csv"
S123 = "cnxb_512_e16_ra_all_s123_test_tta4.csv"
AVG2 = "seed7_s42_test_tta4.csv"
AVG3 = "seed7_s42_s123_test_tta4.csv"
ENS10 = "submission_20251018_ens10_cal.csv"
RA_BEST = "cnxb_512_e16_ra_test_best_tta4.csv"
RA_LAST = "cnxb_512_e16_ra_test_last_tta4.csv"
SWIN = "swin_pair_test_tta4.csv"
TARGET_987 = "4bf37b7c062f7d45a9c5a8e287fff1eb"


def read_map(p):
    with open(p, encoding="utf-8", newline="") as f:
        return {r["Image_name"]: float(r["Probability"]) for r in csv.DictReader(f)}


def norm_rank(v):
    v = np.asarray(v, dtype=float)
    n = len(v)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
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


def spearman(a, b):
    x = norm_rank(a) - norm_rank(a).mean()
    y = norm_rank(b) - norm_rank(b).mean()
    return float((x * y).sum() / np.sqrt((x ** 2).sum() * (y ** 2).sum()))


def topk(a, b, k=200):
    return len(set(np.argsort(-np.asarray(a))[:k].tolist()) & set(np.argsort(-np.asarray(b))[:k].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "audit_185", "check_seed123.json"))
    a = ap.parse_args()

    names = [r["Image_name"] for r in csv.DictReader(open(os.path.join(ROOT, "glaucoma_first", "test.csv"),
                                                          encoding="utf-8", newline=""))]
    assert len(names) == 5855

    files = {"seed7": S7, "seed42": S42, "seed123": S123, "avg2": AVG2, "avg3": AVG3,
             "ens10": ENS10, "ra_best": RA_BEST, "ra_last": RA_LAST, "swin_pair": SWIN}
    vec, member, hashes = {}, {}, {}
    for k, fn in files.items():
        p = os.path.join(ROOT, fn)
        d = read_map(p)
        blob = open(p, "rb").read()
        hashes[k] = {"file": fn, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
                     "md5": hashlib.md5(blob).hexdigest()}
        rec = {"rows": len(d), "official_order": list(d) == names,
               "nan": 0, "out_of_range": 0}
        v = np.array([d[n] for n in names], dtype=float)
        rec["nan"] = int(np.isnan(v).sum())
        rec["out_of_range"] = int(((v < 0) | (v > 1)).sum())
        rec["mean"] = round(float(v.mean()), 6)
        rec["pos_at_0.5"] = int((v >= 0.5).sum())
        rec["min"] = round(float(v.min()), 6)
        rec["max"] = round(float(v.max()), 6)
        if rec["rows"] != 5855 or not rec["official_order"] or rec["nan"] or rec["out_of_range"]:
            sys.exit("ERROR: %s 校验失败 %s" % (fn, rec))
        member[k] = rec
        vec[k] = v

    members = ["seed7", "seed42", "seed123", "avg2", "avg3"]
    pair = {}
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            x, y = members[i], members[j]
            pair["%s vs %s" % (x, y)] = {
                "spearman": round(spearman(vec[x], vec[y]), 6),
                "top200": topk(vec[x], vec[y]),
                "max_absdiff": round(float(np.abs(vec[x] - vec[y]).max()), 6),
            }

    def blend(seedavg_key):
        s = np.zeros(len(names))
        s += 0.60 * norm_rank(vec["ens10"])
        s += 0.10 * norm_rank(vec[seedavg_key])
        s += 0.15 * (0.5 * norm_rank(vec["ra_best"]) + 0.5 * norm_rank(vec["ra_last"]))
        s += 0.15 * norm_rank(vec["swin_pair"])
        return s

    def to_csv(score):
        lines = ["Image_name,Probability,Pred_Label"]
        for n, v in zip(names, score):
            lines.append("%s,%.6f,%d" % (n, v, 1 if v >= 0.5 else 0))
        blob = ("\r\n".join(lines) + "\r\n").encode("utf-8")
        return hashlib.md5(blob).hexdigest()

    b2, b3 = blend("avg2"), blend("avg3")
    t2 = set(np.argsort(-b2)[:200].tolist())
    t3 = set(np.argsort(-b3)[:200].tolist())
    out = {
        "issue": 185,
        "delivered_by": "PR #155（#152）—— 本脚本只做 CPU 侧复核，不重跑训练",
        "members": member,
        "pairwise": pair,
        "blend_98_87_with_2seed_vs_3seed": {
            "spearman": round(spearman(b2, b3), 6),
            "top200": topk(b2, b3),
            "top200_changed": len(t2 ^ t3) // 2,
            "max_absdiff": round(float(np.abs(b2 - b3).max()), 6),
            "md5_2seed": to_csv(b2),
            "md5_3seed": to_csv(b3),
            "md5_2seed_matches_98_87": to_csv(b2) == TARGET_987,
        },
        "hashes": hashes,
        "not_verified": {
            "check_train_on_all": "需要 runs/cnxb_512_e16_ra/last.pt + cache_all768（GPU/数十 GB），本机未执行；"
                                  "上方 pairwise 为其 CPU 替代口径（Spearman/top200/mean 漂移）",
            "valid_auroc": "train+all 无干净 valid，按工单不报",
        },
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: out[k] for k in ("pairwise", "blend_98_87_with_2seed_vs_3seed")},
                     ensure_ascii=False, indent=2))
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
