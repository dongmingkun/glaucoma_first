# Issue #128 结果记录 — Base@512 16ep + RandAugment 2 9（新多样性成员，目标 valid TTA4 ≥0.9925）

> 对应 https://github.com/dongmingkun/glaucoma_first/issues/128 。真机执行完成（2026-10-24），
> 本文件为回填后的完整结果。指标工具：`tools/oof_report.py`（CPU）。

## 配置

| 项 | 值 |
|---|---|
| 命令 | `TAG=cnxb_512_e16_ra SEED=7 RA="2 9" BS=8 ACCUM=6 WORKERS=8 CACHE=cache768 bash tools/pipeline_117.sh` |
| 等价单步 | `train.py --arch convnext_base --img-size 512 --cache-root cache768 --epochs 16 --batch-size 8 --accum 6 --lr 1e-4 --head-lr 5e-4 --pretrained 1 --randaugment 2 9 --loader workers --workers 8 --seed 7 --tag cnxb_512_e16_ra` |
| 机器 | AutoDL RTX 4090 24GB（208 核 CPU / 1TB 内存），不占队长本地 5060 |
| 耗时 | 16:46 → 19:42 ≈ 2h56m（训练 16ep ≈ 2h48m，~10.5min/ep，含 TTA4 推理与校验） |
| 缓存 | `cache768 train=51396 valid=5849 test=5855`（脚本硬校验通过） |
| 初始化 | ImageNet（`--pretrained 1`），seed 7，无热启动/外部医学权重 |
| 历史证据 | RandAugment 2 9 是唯一有效训练侧增强：Base@384 6ep 复验 valid +0.00118；seed7/123 单模 0.99119/0.99154 |

## 每 epoch 曲线（train_loss / val_loss / AUROC / EMA_AUROC / EMA_AUPRC）

完整数据：仓库根目录 `cnxb_512_e16_ra_log.csv`（即 `runs/cnxb_512_e16_ra/log.csv`）。

```
 1: 0.401/0.167/0.96884/0.97521/0.68984
 2: 0.260/0.140/0.97703/0.98223/0.77911
 3: 0.215/0.140/0.97512/0.98452/0.83016
 4: 0.197/0.149/0.98523/0.98734/0.83511
 5: 0.174/0.120/0.98199/0.98786/0.85179
 6: 0.174/0.124/0.98850/0.98963/0.85584
 7: 0.159/0.102/0.98874/0.99093/0.86069
 8: 0.146/0.107/0.98670/0.99087/0.86971
 9: 0.151/0.107/0.99023/0.99085/0.87137
10: 0.130/0.098/0.99046/0.99178/0.87548
11: 0.126/0.096/0.99128/0.99227/0.87731
12: 0.117/0.102/0.99197/0.99268/0.87819  ← EMA 峰值，best.pt（blend 0.9354）
13: 0.112/0.092/0.99109/0.99217/0.87777
14: 0.105/0.110/0.99067/0.99131/0.87731
15: 0.105/0.096/0.99068/0.99110/0.87667
16: 0.102/0.097/0.99033/0.99079/0.87644
```

## best / last(EMA) 两口径 valid / test TTA4 ✅ 双口径达标

```bash
python tools/oof_report.py metrics --labels glaucoma_first/valid.csv \
  --probs cnxb_512_e16_ra_valid_best_tta4.csv cnxb_512_e16_ra_valid_last_tta4.csv
```

| 口径 | valid AUROC | valid AUPRC | balACC@0.5 | max-balACC | test mean | 达标 ≥0.9925 |
|---|---|---|---|---|---|---|
| best（ep12，blend 选中；推理 use=ema） | **0.99322** | **0.88270** | 0.92531 | 0.96814（@p≥0.1295） | 0.05514 | ✅ |
| last + 固定 EMA | **0.99284** | 0.88231 | 0.91293 | 0.96667（@p≥0.1028） | 0.05136 | ✅ |

**对照历史：超过 ens9 集成（0.99110）与此前最强单模 Swin-B@384 16ep（0.99233），
为仓库目前最强单模；AUPRC 0.88270 亦超过 Small@512 参考基线 0.87256。**

## 多样性：与既有成员的 Spearman / 融合

实例上可得的近期异构成员（`tools/oof_report.py` 两两 Spearman，valid 5849）：

| 对（括号内为该成员 valid AUROC） | Spearman(best) | Spearman(last/EMA) |
|---|---|---|
| vs Swin-B@384 16ep（0.99233） | 0.74166 | 0.67693 |
| vs Swin-B@384 16ep seed7（0.99189） | 0.77600 | 0.71598 |
| vs Swin-L@384 16ep（0.98732） | 0.70442 | 0.65003 |
| vs ConvNeXt-Base@384 22k（0.98926） | 0.68590 | 0.65614 |
| vs ConvNeXt-Base@384 pw3（0.98919） | 0.85715 | 0.82774 |
| best × last（自身两口径） | 0.95991 | — |

**与全部现有成员 Spearman 0.65–0.86，远低于 0.98 门槛——多样性极佳**
（同架构 Base 成员间通常 >0.95；RA 训练侧增强带来了真正的排序多样性）。

以下三项需 ens9 / Base@512 4ep / Base@384 8ep 的 valid 概率文件（在队长本地/Release，
云机上没有），**待补**；命令已就绪：

```bash
python tools/oof_report.py metrics --labels glaucoma_first/valid.csv \
  --probs cnxb_512_e16_ra_valid_last_tta4.csv ens9_valid_probs_tta4.csv
python tools/oof_report.py blend --labels glaucoma_first/valid.csv \
  --base ens9_valid_probs_tta4.csv --new cnxb_512_e16_ra_valid_last_tta4.csv
```

| 对 | Spearman(valid) |
|---|---|
| vs ens9（验收关注 <0.98） | 待补（文件未到手；由成员 0.65–0.86 推断大概率 ≪0.98） |
| vs Base@512 4ep | 待补 |
| vs Base@384 8ep | 待补 |
| blend vs ens9：best w / Δ / P(Δ>0)（prob & logit） | 待补 |

## 验收（对照 #128）

- [x] valid TTA4 ≥0.9925：best 0.99322 / last 0.99284，**双口径达标**；
- [x] test 5855 行、顺序一致、无 NaN（pipeline 第 [4] 步硬校验 OK；
      test mean best 0.05514 / last 0.05136，与 valid 患病率 3.27% 同量级）；
- [ ] 与 ens9 Spearman <0.98：与现有 5 个成员 0.65–0.86（远优于门槛），ens9 待补文件后确认；
- [x] 未占用队长本地 5060（AutoDL 4090）。

## 交付物

- 权重：`runs/cnxb_512_e16_ra/{best,last}.pt`（云端实例 `/root/glaucoma_first/` 下，
  待挂 Release/群文件）；
- 概率 CSV × 4（本 PR，仓库根目录，命名与流水线产物一致）：
  `cnxb_512_e16_ra_{valid,test}_{best,last}_tta4.csv`（5849/5855 行）；
- 训练曲线：`cnxb_512_e16_ra_log.csv`（仓库根目录，即 `log.csv`）；
- 本文件回填后的完整数字。
