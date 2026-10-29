# -*- coding: utf-8 -*-
"""独立复现 98.87（#186 Task A）—— 纯 CPU、numpy、不读 test 标签。

口径（由 L1 `e244a37e` 反推锁定，见 docs/ISSUE_186_RESULTS.md）：
    score = 0.60*rank(ens10)
          + 0.10*rank(seed7_s42_test_tta4.csv)
          + 0.15*(0.5*rank(RA_best) + 0.5*rank(RA_last))
          + 0.15*rank(swin_pair_test_tta4.csv)

`rank` = 测试集内 (rank-1)/(n-1)（0-based，并列取平均秩）。
RA_avg 是**先各自 rank 再 0.5/0.5 平均**（不重新排名）——
这正是队长流水线与 `tools/rank_blend.py`（先概率平均再 rank）的分歧点，
后者复算 98.87 会得到另一个 md5（#180/#179 已记录）。

输出格式与队长产物一致：CRLF + `Image_name,Probability,Pred_Label` + `%.6f`，
`Pred_Label = score >= 0.5`（不参与评分）。

用法：
  python tools/reproduce_98_87.py --check-md5
  python tools/reproduce_98_87.py --out /tmp/repro_98_87.csv

默认成员路径为仓库根下文件名（官方 test 顺序取 `glaucoma_first/test.csv`）。
"""
import argparse
import csv
import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_MD5 = "4bf37b7c062f7d45a9c5a8e287fff1eb"

DEFAULTS = {
    "ens10": "submission_20251018_ens10_cal.csv",
    "seedavg": "seed7_s42_test_tta4.csv",
    "ra_best": "cnxb_512_e16_ra_test_best_tta4.csv",
    "ra_last": "cnxb_512_e16_ra_test_last_tta4.csv",
    "swin_pair": "swin_pair_test_tta4.csv",
    "test_csv": os.path.join("glaucoma_first", "test.csv"),
}


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


def read_probs(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {r["Image_name"]: float(r["Probability"]) for r in csv.DictReader(f)}


def build(names, paths):
    def col(key):
        d = read_probs(paths[key])
        if list(d) != names:
            sys.exit("ERROR: %s 的名序与官方 test.csv 不一致" % paths[key])
        return np.array([d[n] for n in names], dtype=float)

    score = np.zeros(len(names))
    score += 0.60 * norm_rank(col("ens10"))
    score += 0.10 * norm_rank(col("seedavg"))
    score += 0.15 * (0.5 * norm_rank(col("ra_best")) + 0.5 * norm_rank(col("ra_last")))
    score += 0.15 * norm_rank(col("swin_pair"))
    return score


def to_csv_bytes(names, score):
    lines = ["Image_name,Probability,Pred_Label"]
    for n, v in zip(names, score):
        lines.append("%s,%.6f,%d" % (n, v, 1 if v >= 0.5 else 0))
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="输出 CSV 路径（默认不落盘）")
    ap.add_argument("--check-md5", action="store_true", help="与 98.87 目标 md5 比对")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=os.path.join(ROOT, v))
    a = ap.parse_args()

    paths = {k: getattr(a, k) for k in DEFAULTS}
    for k, p in paths.items():
        if not os.path.exists(p):
            sys.exit("ERROR: 缺文件 %s（%s）" % (p, k))

    with open(paths["test_csv"], encoding="utf-8", newline="") as f:
        names = [r["Image_name"] for r in csv.DictReader(f)]
    if len(names) != 5855:
        sys.exit("ERROR: 官方 test.csv 行数 %d != 5855" % len(names))

    score = build(names, paths)
    blob = to_csv_bytes(names, score)
    md5 = hashlib.md5(blob).hexdigest()
    print("rows=%d  mean=%.6f  range=[%.6f, %.6f]" % (len(names), score.mean(), score.min(), score.max()))
    print("md5=%s" % md5)
    if a.check_md5:
        ok = md5 == TARGET_MD5
        print("target=%s -> %s" % (TARGET_MD5, "MATCH" if ok else "MISMATCH"))
        if not ok:
            return 1
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "wb") as f:
            f.write(blob)
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
