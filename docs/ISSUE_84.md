# Issue #84 — 类平衡：Base@384 × pos_weight=3（阶段1 负结果，归档 / 待队长算增益）

> 状态：阶段1 完成，valid TTA4 0.98919 < 0.990 且 < base@384 0.99147（单变量负结果）｜ 指派人：Yulingling-sc ｜ 执行环境：AutoDL 4090D
> 本文为 issue #84 阶段1 结果的归档；**权威内容以 GitHub issue #84 评论为准**。

---

## 1. 背景

现有 base@384 用默认类权重 pos_weight=5（正类/病变类权重偏低）。本工单只改**类平衡超参**（单变量）：`--pos-weight 3.0`（5→3，相对提升病变类权重），其余配置与 base@384 完全一致。目标：检验类权重调整能否带来质量增益（单变量，不引入其他变化）。

## 2. 训练配置（train-only 8ep）

```bash
python train.py --arch convnext_base --img-size 384 --cache-root cache512   --epochs 8 --batch-size 12 --accum 4 --lr 1e-4 --head-lr 5e-4   --loader workers --workers 3 --seed 42 --pos-weight 3.0 --tag cnxb_384_pw3
```

| 项 | 值 |
|---|---|
| 架构 / 分辨率 | ConvNeXt-Base @384，ImageNet-1k 初始化 |
| 单变量 | 类权重 pos_weight 5→3（其余与 base@384 完全一致） |
| epoch / seed | 8 / 42 |
| 显存 / 耗时 | 6.4G / 每 epoch ≈ 7 min，8ep 约 1h |

## 3. 训练曲线（EMA 口径）

| epoch | val_loss | AUROC(单变) | AUPRC | EMA AUROC | EMA AUPRC |
|---|---|---|---|---|---|
| 1 | 0.1779 | 0.9704 | 0.7404 | 0.9811 | 0.7730 |
| 2 | 0.1340 | 0.9803 | 0.7713 | 0.9827 | 0.8025 |
| 3 | 0.0953 | 0.9866 | 0.7953 | 0.9852 | 0.8327 |
| 4 | 0.0852 | 0.9874 | 0.8341 | 0.9891 | 0.8516 |
| 5 | 0.0767 | 0.9890 | 0.8582 | 0.9888 | 0.8577 |
| 6 | 0.0786 | 0.9854 | 0.8572 | 0.9883 | 0.8651 |
| 7 | 0.0717 | 0.9884 | 0.8712 | 0.9883 | 0.8692 |
| 8 | 0.0711 | 0.9884 | 0.8703 | 0.9886 | 0.8700 |

## 4. 验收

`infer.py --ckpt <best.pt> --split valid --tta 4 --use ema`：

- **valid AUROC = 0.98919 / AUPRC = 0.86424**
- 对照 base@384（默认 pos_weight=5，valid TTA4 **0.99147**）→ pos_weight=3 档 **−0.00228**
- 每 epoch EMA AUROC 在 0.9811→0.9891 间爬升（ep4 峰值 0.9891），ep8 EMA 0.9886

## 5. 交付物（Release `weights-1016`，EMA 口径，与 #76/#79/#83 一致）

| 文件 | 说明 |
|---|---|
| `cnxb_384_pw3_valid_tta4.csv` | 干净 valid TTA4 概率，5849 行 |
| `cnxb_384_pw3_test_probs_tta4.csv` | test TTA4 概率，5855 行，未校准 |

## 6. 结论与下一步

- **单变量负结果**：类权重 5→3（增强病变类权重）**未带来质量增益**，valid TTA4（EMA）0.98919 < 0.990 门槛、且低于 base@384（0.99147）。
- 类平衡调整在本数据集上未产生正向效果；阶段2（train+all + 伪标签）是否触发取决于队长用干净 valid 离线算出的真实集成增益。
- 两份概率仍挂出，供队长离线算集成增益；若增益为非负再议。
- **初步建议归档为「负结果 / 不进集成」**。
