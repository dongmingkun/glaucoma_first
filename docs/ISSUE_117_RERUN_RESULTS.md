# Issue #125 结果记录 — 云端重跑 #117：ConvNeXt-Base@512 16ep batch8×6（目标 valid TTA4 ≥0.9925）

> 对应 https://github.com/dongmingkun/glaucoma_first/issues/125 。真机执行完成（2026-10-24），
> 本文件为回填后的完整结果。指标工具：`tools/oof_report.py`（CPU）。
> **结论先行：valid TTA4 未达 0.9925（best 0.99193 / last 0.98742），按工单约定如实记录负结果；
> best AUPRC 0.88361 为全仓库最高，但比赛指标是 AUROC，且与 ens9 的 blend 没有可用增益
> （best Δ+0.00014 / P=0.737）；与 #128 的对照是 RA+seed 同时变化的非单变量对照，不能单独归因 RA。**

## 配置

| 项 | 值 |
|---|---|
| 命令 | `TAG=cnxb_512_e16_cloud BS=8 ACCUM=6 WORKERS=8 CACHE=cache768 bash tools/pipeline_117.sh` |
| 机器 | AutoDL RTX 4090 24GB（不占队长本地 5060） |
| 耗时 | 19:42 → 22:39 ≈ 2h57m（训练 16ep ≈ 2h47m，~10.4min/ep，含 TTA4 推理与校验） |
| 缓存 | `cache768 train=51396 valid=5849 test=5855`（脚本硬校验通过） |
| 初始化 | ImageNet，seed 42，无 RandAugment、无热启动、无外部医学权重；valid 干净（无 train-on-all） |
| 对照 | #117 本地 5060 8GB batch4×accum12：epoch9 EMA valid AUROC 0.987（未完成 16ep） |

## 每 epoch 曲线（train_loss / val_loss / AUROC / EMA_AUROC / EMA_AUPRC）

完整数据：仓库根目录 `cnxb_512_e16_cloud_log.csv`（即 `runs/cnxb_512_e16_cloud/log.csv`）。

```
 1: 0.363/0.196/0.97638/0.97970/0.76407
 2: 0.217/0.146/0.98214/0.98050/0.80307
 3: 0.186/0.145/0.97215/0.98230/0.82958
 4: 0.162/0.122/0.98442/0.98383/0.84693
 5: 0.156/0.120/0.98300/0.98826/0.85180
 6: 0.136/0.127/0.98358/0.98914/0.86141
 7: 0.130/0.118/0.98249/0.98878/0.86965
 8: 0.118/0.115/0.98612/0.98926/0.87306
 9: 0.110/0.110/0.98293/0.98957/0.87654  ← best.pt（blend 0.9331，EMA AUPRC 峰值）
10: 0.107/0.114/0.98695/0.99007/0.87381  ← EMA AUROC 峰值
11: 0.098/0.115/0.98647/0.98917/0.87233
12: 0.086/0.126/0.98331/0.98782/0.86844
13: 0.081/0.133/0.98477/0.98674/0.86294
14: 0.078/0.126/0.98472/0.98587/0.86210
15: 0.070/0.140/0.98324/0.98480/0.86081
16: 0.071/0.139/0.98358/0.98388/0.85938
```

ep10 后 EMA 缓慢回落（lr 衰减尾部轻微过拟合），best.pt 保留 ep9 权重。

## best / last(EMA) 两口径 valid / test TTA4 ❌ 未达 0.9925（负结果，如实记录）

```bash
python tools/oof_report.py metrics --labels glaucoma_first/valid.csv \
  --probs cnxb_512_e16_cloud_valid_best_tta4.csv cnxb_512_e16_cloud_valid_last_tta4.csv
```

| 口径 | valid AUROC | valid AUPRC | balACC@0.5 | max-balACC | test mean | 达标 ≥0.9925 |
|---|---|---|---|---|---|---|
| best（ep9，blend 选中；推理 use=ema） | 0.99193 | **0.88361** | 0.93216 | 0.96579（@p≥0.1606） | 0.05979 | ❌ 差 0.00057 |
| last + 固定 EMA | 0.98742 | 0.87065 | 0.91022 | 0.95371（@p≥0.0555） | 0.04612 | ❌ |

- 24GB batch8×6（等效 batch 48，与本地 batch4×12 相同）没有把 #117 配方推过 0.9925：
  瓶颈不在显存/批配置，而在配方本身（该配方无 RA 的上限就在 0.992 附近）；
- **best 口径 AUPRC 0.88361 为全仓库最高**（超 #128 的 0.88270 与 Small@512 基线 0.87256），
  但比赛指标是 **AUROC**，AUPRC 高不能直接换算成线上收益；
- **AUROC 0.99193 低于 ens9 0.99260（−0.00067）**，last 0.98742 更低；
- 与 ens9 的 valid blend 只有 best +0.00014 / P=0.737、last +0.00003 / P=0.739
  （w 同批选，已偏乐观）→ **没有 AUROC 集成增益证据，不能据此称“对集成仍有价值”**。

## 与 #128 的对照（非单变量：RA + seed 同时变化；best 均 valid-selected）

| | #125（无 RA，seed42） | #128（RA 2 9，seed7） | Δ |
|---|---|---|---|
| best valid AUROC | 0.99193 | **0.99322** | **+0.00129** |
| best valid AUPRC | **0.88361** | 0.88270 | −0.00091 |
| 两模型 Spearman（best 口径） | 0.77418 | — | 差异大，RA+seed 同时带来排序多样性 |

> 注意：#125 vs #128 同时改了 **RandAugment 和 seed**（两个变量），best 分别为 ep9/ep12 且都是
> valid-selected；不能把 +0.00129 单独归因给 RA。last 口径 #125 0.98742 vs #128 0.99284
> （+0.00542），但 last 是固定 ep16。只能说“RA+seed 组合下 best/last 都更好”，不是干净单变量效应。

与 #117 本地 8GB 的间接对照：本地 ep9 EMA 0.987（batch4×12）→ 云端 ep9 EMA 0.98957
（batch8×6），云端略优但同量级，进一步说明批配置不是决定因素。

## 与既有成员的 Spearman / 融合

实例上可得的近期异构成员（`tools/oof_report.py`，valid 5849）：

| 对（括号内为该成员 valid AUROC） | Spearman(best) | Spearman(last) |
|---|---|---|
| vs #128 cnxb_512_e16_ra（0.99322） | 0.77418 | 0.74550 |
| vs Swin-B@384 16ep（0.99233） | 0.76436 | 0.71254 |
| vs Swin-L@384 16ep（0.98732） | 0.64058 | 0.63356 |
| vs ConvNeXt-Base@384 22k（0.98926） | 0.72474 | 0.74174 |
| best × last（自身两口径） | 0.90566 | — |

ens9 / Base@512 4ep / Base@384 8ep 的 valid 概率已从历史成员重建并核对
（ens9 = 0.591*cb512 + 0.224*hi512_v2 + 0.141*cb384 + 0.044*b512，valid AUROC 0.99260）。
文件在 `release_assets/20261024_team_valid/`（团队 zip）：

```bash
python tools/oof_report.py metrics --labels glaucoma_first/valid.csv \
  --probs ens9_valid_tta4_cal.csv base512_4ep_valid_tta4.csv cb384_valid_tta4.csv \
          cnxb_512_e16_cloud_valid_best_tta4.csv cnxb_512_e16_cloud_valid_last_tta4.csv
python tools/oof_report.py blend --labels glaucoma_first/valid.csv \
  --base ens9_valid_tta4_cal.csv --new cnxb_512_e16_cloud_valid_best_tta4.csv \
  --w-max 0.5 --w-step 0.05 --bootstrap 2000
```

| 对（括号内为该成员 valid AUROC） | Spearman(best) | Spearman(last) |
|---|---|---|
| vs ens9（0.99260） | 0.87396 | 0.81929 |
| vs Base@512 4ep（0.99089） | 0.85011 | 0.79495 |
| vs Base@384 8ep（0.99117） | 0.86447 | 0.81403 |

blend vs ens9（valid 5849；w=0..0.5 step 0.05；2000 次配对 bootstrap）：

| 口径 | mode | best w | blend AUROC | ΔAUROC | P(Δ>0) |
|---|---|---|---|---|---|
| best（valid-selected） | prob | 0.45 | 0.99274 | +0.00014 | 0.737 |
| best（valid-selected） | logit | 0.20 | 0.99272 | +0.00012 | 0.747 |
| last（固定 EMA） | prob | 0.05 | 0.99263 | +0.00003 | 0.739 |
| last（固定 EMA） | logit | 0.00 | 0.99260 | +0.00000 | — |

> w 同批选、P 为条件推断，已偏乐观；即便这样 Δ 也只有 +0.0001 量级、P<0.75。
> **结论：AUROC 上没有可用的集成增益；AUPRC 最高只作记录，不纳入集成。**

## 验收（对照 #125）

- [x] valid/test 行数 5849/5855、顺序与官方 CSV 一致、无 NaN（pipeline 第 [4] 步硬校验 OK）；
- [x] **valid TTA4 <0.9925 → 负结果，如实记录，未包装成正收益**；
- [x] 报告 best 与 last（固定 EMA）两个口径；
- [x] vs ens9 Spearman 0.87396(best)/0.81929(last) <0.98；blend vs ens9 无 AUROC 增益
      （best Δ+0.00014 / P=0.737），**不纳入集成**；
- [x] 未占用队长本地 5060。

## 交付物

- 权重：`runs/cnxb_512_e16_cloud/{best,last}.pt`，已挂
  Release [`cnxb_512_e16_cloud-1024`](https://github.com/dongmingkun/glaucoma_first/releases/tag/cnxb_512_e16_cloud-1024)；
- 概率 CSV × 4（本 PR，仓库根目录）：`cnxb_512_e16_cloud_{valid,test}_{best,last}_tta4.csv`
  （5849/5855 行）；
- 训练曲线：`cnxb_512_e16_cloud_log.csv`（仓库根目录，即 `log.csv`）；
- 本文件回填后的完整数字。
