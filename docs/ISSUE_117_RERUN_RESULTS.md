# Issue #125 结果记录 — 云端重跑 #117：ConvNeXt-Base@512 16ep batch8×6（目标 valid TTA4 ≥0.9925）

> 对应 https://github.com/dongmingkun/glaucoma_first/issues/125 。本文件为**预交付模板**：
> 云端 24GB+ 跑完 `TAG=cnxb_512_e16_cloud BS=8 ACCUM=6 WORKERS=8 CACHE=cache768 bash tools/pipeline_117.sh`
> 后回填，未填处标 `待填`。指标工具：`tools/oof_report.py`（CPU）。

## 配置

| 项 | 值 |
|---|---|
| 命令 | `TAG=cnxb_512_e16_cloud BS=8 ACCUM=6 WORKERS=8 CACHE=cache768 bash tools/pipeline_117.sh` |
| 机器 | 待填（GPU 型号 / 显存） |
| 耗时 | 待填 |
| 缓存 | `cache768 train=51396 valid=5849 test=5855`（脚本硬校验） |
| 初始化 | ImageNet（`--pretrained 1`），seed 42，无热启动、无外部医学权重 |
| 对照 | 本地 5060 8GB batch4×accum12：epoch9 EMA valid AUROC 0.987（#117 预交付记录） |

## 每 epoch 曲线

从 `runs/cnxb_512_e16_cloud/log.csv` 粘贴（epoch / train_loss / val_loss / AUROC / EMA_AUROC）：

```
待填
```

## best / last(EMA) 两口径 valid / test TTA4

```bash
python tools/oof_report.py metrics --labels glaucoma_first/valid.csv \
  --probs oof/cnxb_512_e16_cloud_valid_best_tta4.csv oof/cnxb_512_e16_cloud_valid_last_tta4.csv
python tools/oof_report.py metrics --labels glaucoma_first/test.csv \
  --probs oof/cnxb_512_e16_cloud_test_best_tta4.csv oof/cnxb_512_e16_cloud_test_last_tta4.csv
```
（test 无官方标签，只看 mean/分布；上表主要回填 valid 两行。）

| 口径 | valid AUROC | valid AUPRC | test mean | 备注 |
|---|---|---|---|---|
| best（auto=train.py 选中权重） | 待填 | 待填 | 待填 | |
| last + 固定 EMA | 待填 | 待填 | 待填 | #117 的主口径 |

**达标判定：valid TTA4 ≥0.9925 → 待填（达标 / 未达标如实记录，不包装）。**

## 与既有成员的 Spearman / 融合（需 Base@512 4ep、Base@384 8ep、ens9 的 valid/test 概率 CSV，Release/群文件取）

```bash
python tools/oof_report.py metrics --labels glaucoma_first/valid.csv \
  oof/cnxb_512_e16_cloud_valid_last_tta4.csv cb512_4ep_valid_probs_tta4.csv \
  cb384_8ep_valid_probs_tta4.csv ens9_valid_probs_tta4.csv
python tools/oof_report.py blend --labels glaucoma_first/valid.csv \
  --base ens9_valid_probs_tta4.csv --new oof/cnxb_512_e16_cloud_valid_last_tta4.csv
```

| 对 | Spearman(valid) |
|---|---|
| vs Base@512 4ep | 待填 |
| vs Base@384 8ep | 待填 |
| vs ens9 | 待填 |
| blend vs ens9：best w / Δ / P(Δ>0)（prob & logit） | 待填 |

## 验收（对照 #125）

- [ ] valid/test 行数 5849/5855、顺序与官方 CSV 一致、无 NaN（pipeline 第 [4] 步硬校验）；
- [ ] valid TTA4 ≥0.9925（未达标则如实记录负结果）；
- [ ] 报告 best 与 last（固定 EMA）两个口径；
- [ ] 未占用队长本地 5060。

## 交付物

- `runs/cnxb_512_e16_cloud/{best,last}.pt` — 位置：待填（Release 链接 / 群文件）；
- 4 个概率 CSV：`oof/cnxb_512_e16_cloud_{valid,test}_{best,last}_tta4.csv`；
- 本文件回填后的完整数字。
