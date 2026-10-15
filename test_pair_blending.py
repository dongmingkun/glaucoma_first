"""Test patient-pair (both eyes) probability blending on the valid split.

The dataset is organised in consecutive same-age pairs = the two eyes of one patient
(verified in the audit: labels agree within a pair 96.4% of the time). Glaucoma is
usually bilateral, so blending an image's score with its partner-eye score denoises
the ranking. This uses only the test-set structure (ages), never labels.

Run: python test_pair_blending.py
"""
import csv
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = r"C:\Users\Kevin\Desktop\glaucoma_first"
RUNS = {
    "384": r"D:\glaucoma_work\runs\efficientnet_b3_384_1014_194338_b3_384_cont",
    "448": r"D:\glaucoma_work\runs\efficientnet_b3_448_1014_221755_b3_448",
    "512": r"D:\glaucoma_work\runs\efficientnet_b3_512_1014_233847_b3_512",
    "cx": r"D:\glaucoma_work\runs\convnext_tiny_384_1015_011832_cnxt_384",
    "cs": r"D:\glaucoma_work\runs\convnext_small_384_1015_060006_cnxs_384",
    "cb": r"D:/glaucoma_work/runs/convnext_base_384_1015_113411_cnxb_384",
}

rows = list(csv.DictReader(open(os.path.join(ROOT, "glaucoma_first", "valid.csv"), encoding="utf-8-sig")))
ages = [r["Age"] for r in rows]
y = np.array([int(r["True_Label"]) for r in rows])

probs = {k: np.load(os.path.join(v, "probs_valid_full_tta4.npy")) for k, v in RUNS.items()}
b3 = (probs["384"] + probs["448"] + probs["512"]) / 3
p = (probs["cb"] + probs["cs"] + 0.5 * b3) / 2.5   # best ensemble so far

# build pairs: consecutive same-age
partner = np.full(len(rows), -1)
i = 0
while i < len(rows) - 1:
    if ages[i] != "" and ages[i] == ages[i + 1]:
        partner[i] = i + 1
        partner[i + 1] = i
        i += 2
    else:
        i += 1
n_paired = int((partner >= 0).sum())
print(f"valid n={len(rows)} paired={n_paired} ({n_paired/len(rows)*100:.1f}%)")


def maxbalacc(pv):
    o = np.argsort(-pv)
    ys, ss = y[o], pv[o]
    idx = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
    tps = np.cumsum(ys)[idx]
    fps = np.cumsum(1 - ys)[idx]
    P, N = ys.sum(), len(ys) - ys.sum()
    return float(np.max(tps / P + (1 - fps / N)) / 2)


def metrics(pv, tag):
    a, ap = roc_auc_score(y, pv), average_precision_score(y, pv)
    print(f"{tag:26s} AUROC={a:.5f} AUPRC={ap:.5f} blend={0.5*(a+ap):.5f} maxbal={maxbalacc(pv):.5f}")


metrics(p, "no blending")
for w in (0.1, 0.2, 0.3, 0.4, 0.5):
    pb = p.copy()
    m = partner >= 0
    pb[m] = (1 - w) * p[m] + w * p[partner[m]]
    metrics(pb, f"pair blend w={w}")

# sanity: does the pairing rule hold on the test set too?
trows = list(csv.DictReader(open(os.path.join(ROOT, "glaucoma_first", "test.csv"), encoding="utf-8-sig")))
tages = [r["Age"] for r in trows]
tp = 0
i = 0
while i < len(tages) - 1:
    if tages[i] != "" and tages[i] == tages[i + 1]:
        tp += 2
        i += 2
    else:
        i += 1
print(f"\ntest n={len(trows)} paired={tp} ({tp/len(trows)*100:.1f}%)")