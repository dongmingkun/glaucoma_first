# Issue #128 结果记录 — Base@512 16ep + RandAugment 2 9（新多样性成员，目标 valid TTA4 ≥0.9925）

> 对应 https://github.com/dongmingkun/glaucoma_first/issues/128 。本文件为**预交付模板**：
> 云端 24GB+ 跑完 `TAG=cnxb_512_e16_ra SEED=7 RA="2 9" BS=8 ACCUM=6 WORKERS=8 CACHE=cache768
> bash tools/pipeline_117.sh` 后回填，未填处标 `待填`。指标工具：`tools/oof_report.py`（CPU）。

## 配置

| 项 | 值 |
|---|---|
| 命令 | `TAG=cnxb_512_e16_ra SEED=7 RA="2 9" BS=8 ACCUM=6 WORKERS=8 CACHE=cache768 bash tools/pipeline_117.sh` |
| 等价单步 | `train.py --arch convnext_base --img-size 512 --cache-root cache768 --epochs 16 --batch-size 8 --accum 6 --lr 1e-4 --head-lr 5e-4 --pretrained 1 --randaugment 2 9 --loader workers --workers 8 --seed 7 --tag cnxb_512_e16_ra` |
| 机器 / 耗时 | 待填 |
| 初始化 | ImageNet，seed 7（历史 RandAugment 有效复验用的 seed），无热启动/外部医学权重 |
| 历史证据 | RandAugment 2 9 是唯一有效训练侧增强：Base@384 6ep 复验 valid +0.00118；seed7/123 单模 0.99119/0.99154 |
| 注意 | 不要覆盖 #125 的 `runs/cnxb_512_e16_cloud`（TAG 不同即隔离，pipeline 已防覆盖） |

## 每 epoch 曲线

从 `runs/cnxb_512_e16_ra/log.csv` 粘贴（epoch / train_loss / val_loss / AUROC / EMA_AUROC）：

```
待填
```

## best / last(EMA) 两口径 valid / test TTA4

```bash
python tools/oof_report.py metrics --labels glaucoma_first/valid.csv \
  --probs oof/cnxb_512_e16_ra_valid_best_tta4.csv oof/cnxb_512_e16_ra_valid_last_tta4.csv
```

| 口径 | valid AUROC | valid AUPRC | test mean |
|---|---|---|---|
| best（auto） | 待填 | 待填 | 待填 |
| last + 固定 EMA | 待填 | 待填 | 待填 |

**达标判定：valid TTA4 ≥0.9925 → 待填。**

## 多样性：与既有成员的 Spearman / 融合

（需 ens9、Base@512 4ep、Base@384 8ep 的 valid 概率 CSV，Release/群文件取；#125 已完成时加上 `cnxb_512_e16_cloud`。）

```bash
python tools/oof_report.py metrics --labels glaucoma_first/valid.csv \
  --probs oof/cnxb_512_e16_ra_valid_last_tta4.csv ens9_valid_probs_tta4.csv \
  cb512_4ep_valid_probs_tta4.csv cb384_8ep_valid_probs_tta4.csv
python tools/oof_report.py blend --labels glaucoma_first/valid.csv \
  --base ens9_valid_probs_tta4.csv --new oof/cnxb_512_e16_ra_valid_last_tta4.csv
```

| 对 | Spearman(valid) |
|---|---|
| vs ens9（验收关注 <0.98） | 待填 |
| vs Base@512 4ep | 待填 |
| vs Base@384 8ep | 待填 |
| vs #125 cnxb_512_e16_cloud | 待填 |
| blend vs ens9：best w / Δ / P(Δ>0)（prob & logit） | 待填 |

## 验收（对照 #128）

- [ ] valid TTA4 ≥0.9925；
- [ ] test 5855 行、顺序一致、无 NaN（pipeline 第 [4] 步硬校验）；
- [ ] 与 ens9 Spearman <0.98 → 优先作为多样性成员；若 ≥0.9925 但相关度高，如实报告两者；
- [ ] 若 <0.9925，记录负结果，不包装成正收益；
- [ ] 未占用队长本地 5060。

## 交付物

- `runs/cnxb_512_e16_ra/{best,last}.pt` — 位置：待填（Release 链接 / 群文件）；
- 4 个概率 CSV：`oof/cnxb_512_e16_ra_{valid,test}_{best,last}_tta4.csv`；
- 本文件回填后的完整数字。
