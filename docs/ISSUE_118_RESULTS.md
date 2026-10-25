# Issue #139 结果记录 — Swin-B@384 16ep × 3-fold clean OOF + test（#127 异构成员）

> 对应 https://github.com/dongmingkun/glaucoma_first/issues/139 （#118 的实际执行）。
> 本文件为**预交付模板**，云端跑完回填，未填处标 `待填`。
> 门槛建议（#127）：能提分 Δ≥+0.0005 且 P≥0.90；99 Δ≥+0.0015 且 P≥0.95。
> 红线：OOF/test 一律 `last.pt + --use ema`（pipeline_118 已含 PR #131/#136 修复）。

## 配置

| 项 | 值 |
|---|---|
| 命令 | `CACHE_SRC=cache512 BS=8 ACCUM=6 bash tools/pipeline_118.sh` |
| 机器 / 耗时 | 待填（4090 24GB，工单估 12–18h） |
| 缓存 | `cache512 51396/5849/5855` → `cache_all` 57245（符号链接，脚本硬校验） |

## 每折 held-out valid TTA4（last.pt + 固定 EMA）

| fold | valid 行数 | 正例 | AUROC | AUPRC | 耗时 |
|---|---|---|---|---|---|
| 0 | 19082 | 616 | 待填 | 待填 | 待填 |
| 1 | 19082 | 616 | 待填 | 待填 | 待填 |
| 2 | 19081 | 616 | 待填 | 待填 | 待填 |

## OOF 57245 整体（#127 clean stack 的 Swin 成员）

| 指标 | 值 |
|---|---|
| OOF AUROC | 待填 |
| OOF AUPRC | 待填 |
| 覆盖 | 57245/57245（脚本硬校验） |

test 3 折平均：`oof/test_probs_tta4.csv`（5855），test_mean = 待填。

## 3 折两两 Spearman（验收 <0.98）

| 对 | Spearman |
|---|---|
| fold0 × fold1 | 待填 |
| fold0 × fold2 | 待填 |
| fold1 × fold2 | 待填 |

## vs ens9 blend + 配对 bootstrap

ens9 valid AUROC 0.99260（0.99260 可从团队 zip 取 ens9_valid_probs_tta4.csv 后计算）：

```bash
python tools/oof_report.py blend --labels glaucoma_first/valid.csv \
  --base ens9_valid_probs_tta4.csv --new oof/swin_valid_subset_tta4.csv
```

| 项 | 值 |
|---|---|
| Spearman(Swin OOF valid 段, ens9) | 待填 |
| blend best w / Δ / P(Δ>0)（prob & logit） | 待填 |

## 验收（对照 #139）

- [ ] OOF 覆盖 57245、3 折无重叠；OOF/test 固定 EMA；test 5855 行/顺序/无 NaN；
- [ ] 3 折两两 Spearman <0.98；
- [ ] 作为 #127 clean stack 成员（不单独宣布提分，由 #127 nested CV gate 决定）。

## 交付物

- `runs/swin_fold{0,1,2}/last.pt` → Release（待挂）；
- `oof/swin_fold{k}_{valid,test}_tta4.csv` × 6 + `oof/oof_probs.npy` + `oof/test_probs_tta4.csv`；
- 训练曲线 × 3（log.csv）；
- 本文件回填。
