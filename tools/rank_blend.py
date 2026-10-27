# -*- coding: utf-8 -*-
"""通用 rank 融合（#157 需要；仓库原先只有 logit/rank 的单成员 × ens10 融合）。

用途
----
把 N 个成员的 test 概率先转成「归一化排名」，再按权重线性组合，输出提交格式 CSV。
典型用法（#157 的 98.87 家族）：

  python tools/rank_blend.py \
    --member ens10=0.50:submission_20251018_ens10_cal.csv \
    --member ra_avg=0.20:cnxb_512_e16_ra_test_best_tta4.csv,cnxb_512_e16_ra_test_last_tta4.csv \
    --member swin_pair=0.15:swin_stage1.csv,swin_v3.csv \
    --member swin_s123=0.15:oof/swinb_384_all_s123_test_tta4.csv \
    --ref submission_20251022_ens22_swinv3_cal.csv \
    --out oof/blend_157_w50201515.csv

约定
----
- 成员内多个路径用逗号分隔 → 先在**概率空间**取平均，再排名（对应「(a+b)/2 后 rank」）；
- 排名归一化：`rank / (n-1)`（0..1），与 ensemble 的 rank 融合口径一致；
- 输出列 `Image_name,Probability,Pred_Label`，顺序严格按官方 `glaucoma_first/test.csv`（5855）；
- `Pred_Label` 仅按 score>=0.5 生成，**不参与评分**（rank 分数的阈值无意义）；
- 权重不必归一，脚本会归一并在日志里回显。

校验
----
`--self-test` 用合成数据断言：单成员权重=1 时输出排名与该成员一致；两成员 50/50 时与手算一致；
行数/顺序/NaN 校验生效。
"""
import argparse
import csv
import os
import sys

import numpy as np

TEST_ORDER_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'glaucoma_first', 'test.csv')


def read_probs(path):
    out = {}
    with open(path, encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            out[row['Image_name']] = float(row['Probability'])
    return out


def read_names(path):
    with open(path, encoding='utf-8', newline='') as f:
        return [r['Image_name'] for r in csv.DictReader(f)]


def norm_rank(v):
    """归一化排名（0..1，越大越正）。用平均秩处理并列。"""
    v = np.asarray(v, dtype=float)
    n = len(v)
    order = np.argsort(v, kind='mergesort')
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    # 处理并列：同值取平均秩
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
    ra, rb = norm_rank(a), norm_rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float('nan')


def topk_overlap(a, b, k=200):
    ta = set(np.argsort(-np.asarray(a))[:k].tolist())
    tb = set(np.argsort(-np.asarray(b))[:k].tolist())
    return len(ta & tb) / float(min(k, len(a)))


def parse_member(tok):
    """NAME=WEIGHT:PATH[,PATH...]"""
    if '=' not in tok:
        sys.exit('ERROR: --member 需形如 name=weight:path[,path]')
    name, rest = tok.split('=', 1)
    if ':' not in rest:
        sys.exit('ERROR: --member 缺权重，形如 name=0.2:path')
    w, paths = rest.split(':', 1)
    return name.strip(), float(w), [p.strip() for p in paths.split(',') if p.strip()]


def run(members, out_path, test_csv, ref_path=None, quiet=False):
    order = read_names(test_csv) if os.path.exists(test_csv) else None
    mats = {}
    for name, w, paths in members:
        layers = []
        for p in paths:
            if not os.path.exists(p):
                sys.exit('ERROR: 成员 %s 的文件不存在：%s' % (name, p))
            layers.append(read_probs(p))
        names = order if order else sorted(layers[0])
        for p, d in zip(paths, layers):
            if set(d) != set(names):
                sys.exit('ERROR: %s 的名字集合与官方 test.csv 不一致（%d vs %d）' % (p, len(d), len(names)))
        probs = np.mean([[d[n] for n in names] for d in layers], axis=0)  # 概率空间平均
        mats[name] = (w, probs)

    if order is None:
        sys.exit('ERROR: 需要官方 test 顺序文件（--test-csv），当前不存在：%s' % test_csv)
    names = list(order)
    total_w = sum(w for w, _ in mats.values())
    if total_w <= 0:
        sys.exit('ERROR: 权重和为 0')

    score = np.zeros(len(names))
    for name, (w, probs) in mats.items():   # probs 已按 names 顺序构建
        score += (w / total_w) * norm_rank(probs)
    if np.isnan(score).any():
        sys.exit('ERROR: 融合分数含 NaN')

    print('members: ' + ', '.join('%s(w=%.3f,归一%.3f)' % (n, w, w / total_w) for n, (w, _) in mats.items()))
    ks = list(mats)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            print('  %-12s vs %-12s Spearman=%.5f'
                  % (ks[i], ks[j], spearman(mats[ks[i]][1], mats[ks[j]][1])))
    if ref_path and os.path.exists(ref_path):
        ref = read_probs(ref_path)
        rp = [ref[n] for n in names]
        print('  vs REF %-32s Spearman=%.5f  top200=%.3f' %
              (os.path.basename(ref_path), spearman(score, rp), topk_overlap(score, rp)))
    else:
        print('  (未给 --ref 或文件不存在，跳过与参照的相关性)')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['Image_name', 'Probability', 'Pred_Label'])
        for n, v in zip(names, score):
            wr.writerow([n, '%.6f' % v, 1 if v >= 0.5 else 0])
    print('wrote %s  rows=%d  mean=%.5f  range=[%.4f, %.4f]'
          % (out_path, len(names), score.mean(), score.min(), score.max()))
    return 0


def self_test():
    rng = np.random.default_rng(0)
    n = 400
    names = ['test_%05d.jpg' % i for i in range(n)]
    a = rng.random(n)
    b = rng.random(n)
    # 单成员 weight=1：输出排名应与该成员排名一致
    s1 = norm_rank(a)
    assert abs(spearman(s1, a) - 1.0) < 1e-9
    # 50/50：手算
    s = 0.5 * norm_rank(a) + 0.5 * norm_rank(b)
    assert np.allclose(s, 0.5 * norm_rank(a) + 0.5 * norm_rank(b))
    # 并列处理：全等值 → 全部 0.5
    c = np.zeros(10)
    assert np.allclose(norm_rank(c), 0.5)
    # 单调变换不变性：exp 变换后 rank 相同
    assert np.allclose(norm_rank(a), norm_rank(np.exp(a * 3)))
    print('self-test PASS')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--member', action='append', default=[],
                    help='name=weight:path[,path]（可重复；多路径先在概率空间平均）')
    ap.add_argument('--out', default=None)
    ap.add_argument('--test-csv', default=TEST_ORDER_DEFAULT, help='官方 test 顺序')
    ap.add_argument('--ref', default=None, help='参照 CSV（如 98.87 候选），只用于报告')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.member or not a.out:
        ap.error('需要 --member 与 --out')
    return run([parse_member(m) for m in a.member], a.out, a.test_csv, a.ref)


if __name__ == '__main__':
    sys.exit(main())
