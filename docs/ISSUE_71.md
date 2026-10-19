# Issue #71 — 容量上探：ConvNeXt-Large@384（阶段1 负结果收手）

> 状态：阶段1 负结果（valid < 0.9915，不进入阶段2）｜ 指派人：Yulingling-sc ｜ 执行环境：AutoDL 4090D
> 本文为 issue #71 阶段1 结果的归档；**权威内容以 GitHub issue #71 评论为准**。

---

## 1. 背景

当前最强单模 = convnext_base@384（valid 0.99147）。本工单上探**容量**：convnext_large（198M，base 的 2.2 倍）能否带来单模收益。先跑 train-only 过 0.9915 门槛，再决定是否叠加 train+all + 伪标签。

## 2. 训练配置（从零 train-only）

```bash
python train.py --arch convnext_large --img-size 384 --cache-root cache512 \
  --epochs 7 --batch-size 5 --accum 7 --lr 1e-4 --head-lr 5e-4 \
  --loader workers --workers 3 --seed 42 --tag cnxl_384
```

| 项 | 值 |
|---|---|
| 架构 / 分辨率 | convnext_large @ 384（cache512，198M 参数） |
| 训练方式 | 从零训练（无 --init-from），lr 1e-4 |
| epoch | 7（train-only，valid 干净可评估） |
| 显存 / 耗时 | 6.7G（未 OOM）/ 每 epoch ≈ 15 min，7ep 约 105 min |

（前置：train.py ARCH_INFO 增加 convnext_large，PR #72。）

## 3. 训练曲线（valid）

| epoch | AUROC | AUPRC | EMA AUROC | EMA AUPRC |
|---|---|---|---|---|
| 2 | 0.9818 | 0.7560 | **0.9873** | 0.8135 |
| 3 | 0.9799 | 0.7690 | 0.9863 | 0.8157 |
| 4 | 0.9797 | 0.8023 | 0.9864 | 0.8233 |
| 5 | **0.9865** | 0.8281 | 0.9862 | 0.8173 |
| 6 | 0.9796 | 0.7811 | 0.9831 | 0.8133 |
| 7 | 0.9813 | 0.7970 | 0.9827 | 0.8140 |

train_loss 单调下降（→0.1063），但 val_loss 在 ep5 后回升（0.1264→0.1582→0.1442），7ep 已开始过拟合。

## 4. 结论

**峰值 EMA AUROC = 0.9873（ep2）/ 普通 0.9865（ep5），均 < 0.9915 门槛**（也低于 base@384 的 0.99147）。

按工单约定记负收手，不进入阶段2。容量收益被从零训练的欠拟合 + 后期过拟合抵消。

可选后续（未执行）：更短 epoch（5ep）+ 更强正则，或热启动 base 权重微调——但本次按约定收手。

## 5. 交付物

- 无有效交付物（未达门槛）。
- 训练产物保留在云 `runs/cnxl_384/`，权重不入库。
