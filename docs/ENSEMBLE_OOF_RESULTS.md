# ENSEMBLE OOF 结果（#127）— clean stack 门槛判定

> 对应 https://github.com/dongmingkun/glaucoma_first/issues/127 。本文件为**二成员初步版**
> （cnxs3 + ra3；swin3 待 #139 队友产物后补跑三成员终版）。全部成员均为 fold-trained +
> `last.pt` 固定 EMA（#126/#134，PR #131/#136 口径），无任何全局单模 OOF、无 valid-selected。
> 工具：`tools/stack_oof.py`（nested 3-fold CV，内层拟合/选参，外层 held-out 评估）。

## 结果（2026-10-25，AutoDL 4090 CPU 侧）

| 项 | 值 |
|---|---|
| 成员 | base = cnxs3（#126，OOF AUROC 0.98685）+ ra3 = #134（OOF AUROC 0.98887） |
| 成员间 Spearman | 0.66772（57245 全量，互补性极佳） |
| 基线（--ens9 槽位） | base3 = cnxs3 单独（干净 fold baseline，runbook 认可口径） |
| **选中方法** | **logreg**（logit 空间 L2 逻辑回归，λ=3 内层 CV 选出；coef：cnxs3 0.779 / ra3 1.497） |
| **CV-OOF AUROC** | **0.98929**（rank 0.98895 / weight 0.98904 / logreg 0.98929） |
| CV-OOF AUPRC | 0.85334 |

## 门槛判定（配对 bootstrap，B=2000，seed 42）

| 判定 | 要求 | 实测 | 结论 |
|---|---|---|---|
| 相对 cnxs3 baseline 初筛 | Δ≥+0.0003 且 P≥0.90 | Δ=+0.00244 / P=1.000 | PASS（仅相对弱 base） |
| 相对 cnxs3 baseline 目标 | Δ≥+0.0015 且 P≥0.95 | Δ=+0.00244 / P=1.000（CI95 [+0.00124, +0.00375]） | PASS（仅相对弱 base） |
| **绝对 vs ens9** | stack OOF ≥ ens9 0.99260 | stack **0.98929** | **FAIL（−0.00331）** |
| **线上提分/上99** | 需有 vs ens9/线上 的证据 | **无** | **不能判定；不作候选** |

> **结论：本二成员 stack 不是提分/99 候选，`stack2_test_tta4.csv` 不提交。**
> “相对 cnxs3 +0.00244/P=1.0”只说明两个弱 fold 成员能互补，不代表能打过 ens9；
> 绝对 OOF 0.98929 远低于 ens9 0.99260。工具 JSON 里的 `target99_pass` 是相对
> weak base3 的标签，不能当线上 99。

## 产物

- `oof/stack2_test_tta4.csv`（5855，logreg stack 的 test 概率；本 PR 交付，mean 0.11556）；
- `oof/stack2_oof_probs.npy`（57245）+ `oof/stack2_weights.json`（方法/系数/门槛判定全记录）
  → Release [`stack2-1025`](https://github.com/dongmingkun/glaucoma_first/releases/tag/stack2-1025)
  （*.npy/*.json 大文件不入库规矩）。

## 重要口径说明（如实）

1. **Δ 是相对干净 fold baseline（cnxs3），不等价于 vs 线上 ens9/ens10（98.82）**——线上基线
   含 4 个全局单模成员，无法进入 57245 OOF。相对 weak base 的 PASS **不能**当 99/提分证据；
   绝对 OOF 0.98929 仍低于 ens9 0.99260；
2. stack test 输出是 logit 空间的原始分数（mean 0.11556，未校准）；**本二成员 version 不作
   提交候选**（即使校准，绝对 OOF 也低于 ens9）；
3. **swin3（#139，队友执行中）到位后跑三成员终版 stack**——是否提分/99 以**绝对 OOF vs ens9
   0.99260**（以及必要时的线上反馈）为准，不是“相对 base 只强不弱”；若三成员绝对 OOF
   仍低于 ens9，则 clean 路径判负。

## 复现

```bash
python tools/stack_oof.py \
  --labels glaucoma_first/train.csv glaucoma_first/valid.csv \
  --folds folds3 --test-names glaucoma_first/test.csv \
  --base "oof/oof_probs.npy::oof/test_probs_tta4.csv" \
  --member "ra3=oof/ra_oof.npy::oof/ra_test_tta4.csv" \
  --ens9 "base3=oof/oof_probs.npy::oof/test_probs_tta4.csv" \
  --out-prefix oof/stack2 --bootstrap 2000 --seed 42
```
