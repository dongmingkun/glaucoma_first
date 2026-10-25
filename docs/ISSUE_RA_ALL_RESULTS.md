# Issue #138 结果记录 — Base@512 16ep + RA 的 train+all 57245 家族成员（test-only 探索）

> 对应 https://github.com/dongmingkun/glaucoma_first/issues/138 。真机执行完成（2026-10-25
> 14:47 → 17:50，train/test TTA4/fusion 均 rc=0；check 首跑因工具硬编码路径崩溃，修复
> PR 后重跑完成）。指标工具：`check_train_on_all.py`（--cache-root 修复版）+ `tools/fusion_ens10.py`。
> **声明：本成员 train+all（57245 全量训练），没有干净 valid；只用 test 反馈
> （check_train_on_all + ens10 融合候选），不声称 clean valid 增益；
> 最终是否提分由一次线上提交回答，失败不降低 98.82。**

## 配置

| 项 | 值 |
|---|---|
| 命令 | `train.py --arch convnext_base --img-size 512 --cache-root cache_all768 --train-on-all --epochs 16 --batch-size 8 --accum 6 --lr 1e-4 --head-lr 5e-4 --pretrained 1 --randaugment 2 9 --seed 7 --tag cnxb_512_e16_ra_all` |
| 机器 / 耗时 | AutoDL RTX 4090 24GB；训练 14:47 → 17:48 ≈ 3h01m（16ep，57245 行 ×1.114 步数）；test TTA4 ≈ 2m |
| 数据 | `cache_all768` 57245/57245/5855（#134 建好复用；train+valid 合并 seed 洗牌） |
| 对照 | `runs/cnxb_512_e16_ra/last.pt`（#128 train-only，valid TTA4 0.99284） |

## train+all 预检 ✅

- [x] `--train-on-all` 日志：rows 51396 → 57245（+11.4%）（run138.log）；
- [x] 无 valid 指标虚报（`train_on_all=True` 入 ckpt；last.pt=ep16 EMA，use=ema 推理）。

## check_train_on_all（对照 #128 train-only，test 5855 TTA4）⚠️ REVIEW（边界）

| 项 | 实测 | 工具线 | 工单判负线 |
|---|---|---|---|
| Spearman | **0.8850** | ≥0.90 PASS | <0.80 才判负 |
| mean 相对漂移 | **0.036** | ≤0.25 | 正常家族 13–16% 量级 |
| top200 重叠 | **0.950** | — | — |

**RESULT: REVIEW（未 PASS）**：Spearman 0.885 低于工具 0.90 线、高于工单 0.8 判负线；
分布健康（无塌缩、漂移小、head 重叠 0.95）。对照历史：正常 train+all 成员 0.90–0.919 PASS，
被拒的漂移型成员 0.842——本成员处于中间带。**按 #138 原文不判负，但也不宣称 PASS；
是否作为 ens10 家族候选上线，判定权交队长**（一次线上提交即可回答）。

（check 首跑崩溃为工具问题：`check_train_on_all.py` 调 InferDS 未传 cache_root，落到
队长本机硬编码路径；已修 `--cache-root`（分支 fix/138-check-cache-root），云端重跑得出上表。）

## ens10 融合候选（rank/logit × w=0.1/0.2/0.3，6 个 CSV）

基线 = `submission_20251018_ens10_cal.csv`（98.82 线上版，实例留存）：

| 项 | 值 |
|---|---|
| Spearman(RA_all, ens10) | **0.80989**（> 0.80 报警线，贴近） |
| mean 漂移 | 35.9%（对**校准后** ens10 而言；对原始分参照仅 3.6%，见 check——两口径不同，前者含标定差异） |
| top200 重叠 | 0.950 |
| 6 个候选 CSV | `cnxb_512_e16_ra_all_test_tta4_ens10_{rank,logit}_w{10,20,30}.csv`（本 PR） |
| logit w=10/20/30 翻正数 | 420（vs ens10@0.5） |

**caveat（如实）**：logit 融合是「ens10 校准分 × RA_all 原始分」混合，rank 融合不受标度影响；
与 ens10 秩相关 0.81 处于「可用但需谨慎」区间——多样性收益与校准错位并存，w 越大暴露越大。
**候选不自动提交；最终是否提分由一次线上提交回答（失败不降低 98.82）。**

## 验收（对照 #138）

- [x] test 5855 行、官方顺序、无 NaN、[0,1]（`oof/cnxb_512_e16_ra_all_test_tta4.csv`，mean 0.05319）；
- [x] train-on-all 预检 PASS（+11.4% 行数、无 valid 虚报）；check 相关性 0.885（REVIEW，如实）；
- [x] 明确 test-only 家族候选定位（本文件头部声明）。

## 交付物

- 权重：`runs/cnxb_512_e16_ra_all/last.pt`（673MB），Release
  [`ra_all-1025`](https://github.com/dongmingkun/glaucoma_first/releases/tag/ra_all-1025)；
- `oof/cnxb_512_e16_ra_all_test_tta4.csv`（5855）+ 6 个融合候选 CSV（本 PR）；
- `cnxb_512_e16_ra_all_log.csv`（16 epoch 曲线，train-only 损失口径）；
- check 报告（上表）+ 本文件。

## 建议

- 若队长决定试线：优先 logit_w10（最保守，420 张翻标签，暴露最小）；rank 系与 ens10 相关
  0.81 但幅度大，建议不作为首选；
- 本成员与 #127 stack 无关（train+all 不能进 OOF），不影响已 PASS 的二成员/未来三成员 stack。
