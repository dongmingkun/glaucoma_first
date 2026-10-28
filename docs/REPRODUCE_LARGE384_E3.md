# Large384 提交包（e3 / `e3ddd565`）可复现说明

> 回应 #173 审计：此前用公开成员复算 e3 得 `max_err 0.103`，根因是 **Swin_pair 的两个成员概率文件从未入库**，不是配方或权重问题。本 PR 补齐这两个文件后，e3 可 **100% 复现（max_err 0.000000）**。

## 1. 结论

| 项 | 值 |
|---|---|
| 目标文件 | `submit_161_v4l_20261028/submission.csv` = `oof/blend161_v4l_w35302015.csv` |
| md5 | `e3ddd565dfe15ae6fceb9dfd1c2e172a` |
| 复现误差 | **max_err 0.000000**（5,855 行逐行比对） |
| 复现命令 | 见第 3 节 |

## 2. e3 的四个成员（全部公开）

| 成员 | 权重 | 文件 | md5（前 8） |
|---|---|---|---|
| ens10 | 0.35 | `submission_20251018_ens10_cal.csv` | 已在库 |
| ra_avg | 0.30 | `cnxb_512_e16_ra_test_best_tta4.csv` + `cnxb_512_e16_ra_test_last_tta4.csv` | 已在库 |
| **swin_pair** | 0.20 | **`oof/swinb_384_e16_test_probs_tta4.csv`** + **`oof/swinb_384_all_v3_test_probs_tta4.csv`** | `0a774bfd` / `23262b6f`（**本 PR 新增**） |
| cnxl（Large384） | 0.15 | `oof/cnxl_384_e16_ra_all_s123_test_tta4.csv` | 已在库（#169） |

两个新增文件的校验：均为 5,855 行、官方顺序、表头 `Image_name,Probability,Pred_Label`、无 NaN；
mean 分别 0.05934 / 0.05561（未校准原始概率）。

## 3. 复现命令

```bash
python tools/rank_blend.py \
  --member ens10=0.35:submission_20251018_ens10_cal.csv \
  --member ra_avg=0.30:cnxb_512_e16_ra_test_best_tta4.csv,cnxb_512_e16_ra_test_last_tta4.csv \
  --member swin_pair=0.20:oof/swinb_384_e16_test_probs_tta4.csv,oof/swinb_384_all_v3_test_probs_tta4.csv \
  --member cnxl=0.15:oof/cnxl_384_e16_ra_all_s123_test_tta4.csv \
  --out /tmp/e3_repro.csv
```

逐行比对 `oof/blend161_v4l_w35302015.csv`：`max_err 0.000000`。

## 4. 为什么之前算不准（审计归因）

用仓库里现成的 `swin_pair_test_tta4.csv`（#160 官方）替代本单的 swin_pair，会得到：

| 对比 | max_err | 说明 |
|---|---|---|
| e3 vs 用官方 swin_pair 复算 | **0.1026** | 与审计报出的 0.103 一致 |
| 本单 swin_pair vs 官方 swin_pair | 0.2815 | 两者不是同一份东西：官方 pair 由 stage1+s7 概率平均（mean 0.06011），本单是 `swinb_384_e16`（best）+ `swinb_384_all_v3`（stage1 train+all）平均（mean 0.05747） |
| 但 top200 重叠 | **198 / 200** | 排序几乎一致，概率尺度差异不影响头部结论 |

即：**差异来自成员版本口径，不是配方走样**；对榜单的影响上限是 2 个头部名额。

## 5. 与 L1 的关系

- 队长重建的 L1（md5 `e244a37e`，Release `l1-large384-1028`）用官方 `swin_pair_test_tta4.csv`，与 e3 的 rho 为 0.99747；
- 本 PR 不改变 e3，也不改变 L1 的地位：**L1 仍是口径最干净的一版**；本 PR 只是让 e3 这条线也有据可查、可复现，便于追溯与审计对照；
- 之后的候选（含 #167）如需完全对齐公开口径，建议直接用官方 `swin_pair_test_tta4.csv`。
