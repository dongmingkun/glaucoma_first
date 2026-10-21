# 工单 #95：[P2][训练] 容量上探：Swin-L@384 16ep（195M）

> 状态：已完成（交付概率 + 训练曲线）｜ 指派人：Yulingling-sc ｜ 执行环境：AutoDL 4090D

## 训练配置

```bash
python train.py --arch swin_large --img-size 384 --cache-root cache512 \
  --epochs 16 --batch-size 4 --accum 12 --lr 7e-5 --head-lr 3.5e-4 \
  --loader workers --workers 3 --seed 42 --tag swinl_384_e16
```

> `swinl_384_e16` 是 run tag，`--arch` 取值是 `swin_large`（`timm:swin_large_patch4_window12_384`）。
> 云上 HF 直连不通时用 `HF_ENDPOINT=https://hf-mirror.com`。

## 训练曲线（每 epoch valid AUROC / EMA AUROC / val_loss）

| epoch | auroc | auroc_ema | val_loss |
|---|---|---|---|
| 1 | 0.92209 | 0.96126 | 0.36889 |
| 2 | 0.95291 | 0.9709 | 0.24659 |
| 3 | 0.91441 | 0.97973 | 0.20833 |
| 4 | 0.95275 | 0.97647 | 0.18864 |
| 5 | 0.9761 | 0.98198 | 0.17541 |
| 6 | 0.98054 | 0.9834 | 0.15418 |
| 7 | 0.88903 | 0.98394 | 0.14668 |
| 8 | 0.97077 | 0.98386 | 0.1348 |
| 9 | 0.9855 | 0.98511 | 0.12684 |
| 10 | 0.96302 | 0.98469 | 0.11818 |
| 11 | 0.98586 | 0.98232 | 0.10891 |
| 12 | 0.9441 | 0.98461 | 0.09568 |
| 13 | 0.98721 | 0.98426 | 0.08864 |
| 14 | 0.97384 | 0.98264 | 0.08104 |
| 15 | 0.97635 | 0.98038 | 0.07541 |
| 16 | 0.98161 | 0.98061 | 0.07692 |

## 交付概率（Release `weights-1016`）

| 文件 | 行数 | 口径 |
|---|---|---|
| `swinl_384_e16_valid_tta4.csv` | 5849 | valid TTA4（ema，未校准，集成用） |
| `swinl_384_e16_test_probs_tta4.csv` | 5855 | test TTA4（ema，未校准，集成用） |

## 结论

- valid TTA4 AUROC（ema） = **0.98732**，test TTA4 mean = 0.04517。
- 门槛 valid TTA4 ≥ 0.990：未达标（仍交付概率，增益由队长判定）。
- 真值增益请队长离线并入集成后判定。

Swin-Large@384 容量上探（Swin-B 的 2.2 倍），16ep / lr7e-5 保守档。评估采用 {best,last}x{ema,raw} 四口径 valid TTA4 扫描取最优（#90 实测单点口径会漏分约 0.0016）。无论分数高低均交付概率，增益由队长判定。未跑阶段2（按工单要求）。
