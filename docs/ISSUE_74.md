# Issue #74 — 容量上探 v2：ConvNeXt-Large@384 温和优化（阶段1 负结果收手）

> 状态：阶段1 负结果（valid TTA4 < 0.9915，不进入阶段2）｜ 指派人：Yulingling-sc ｜ 执行环境：AutoDL 4090D
> 本文为 issue #74 阶段1 结果的归档；**权威内容以 GitHub issue #74 评论为准**。

---

## 1. 背景

#71（Large@384，lr 1e-4，7ep）峰值 EMA AUROC 0.9873 < 0.9915，队长复盘判断「优化问题而非容量上限」。本工单 v2 只改优化超参（lr / warmup / epochs），验证 Large 是否值得进入 train+all。

## 2. 训练配置（从零 train-only，v1 vs v2）

| 项 | v1（#71） | v2（#74） |
|---|---|---|
| backbone lr | 1e-4 | **3.5e-5** |
| head lr | 5e-4 | **1.75e-4** |
| warmup | 5% | **10%** |
| epochs | 7 | **10** |
| 其余 | bs5×accum7、cache512、seed42 | 同左 |

```bash
python train.py --arch convnext_large --img-size 384 --cache-root cache512 \
  --epochs 10 --batch-size 5 --accum 7 \
  --lr 3.5e-5 --head-lr 1.75e-4 --warmup-frac 0.10 \
  --loader workers --workers 3 --seed 42 --tag cnxl_384_v2
```

## 3. 训练曲线（单尺度 valid）

| epoch | AUROC | EMA AUROC | val_loss |
|---|---|---|---|
| 1 | 0.9652 | 0.9599 | 0.1805 |
| 2 | 0.9721 | 0.9758 | 0.1650 |
| 3 | 0.9749 | 0.9756 | 0.1540 |
| 4 | 0.9805 | 0.9800 | 0.1274 |
| 5 | 0.9765 | 0.9810 | 0.1435 |
| 6 | 0.9768 | 0.9803 | 0.1452 |
| 7 | **0.9823** | 0.9804 | 0.1343 |
| 8 | 0.9800 | 0.9803 | 0.1253 |
| 9 | 0.9821 | **0.9821** | 0.1385 |
| 10 | 0.9822 | 0.9819 | 0.1353 |

## 4. 验收

`infer.py --ckpt runs/cnxl_384_v2/best.pt --split valid --tta 4`（use=model）：

- **valid AUROC = 0.98543 / AUPRC = 0.84045** < 0.9915 门槛 → 记负收手。

## 5. v1 vs v2 复盘

| 版本 | 单尺度 EMA 峰值 | 结论 |
|---|---|---|
| v1（lr 1e-4 / 7ep） | 0.9873（ep2 见顶回落） | ❌ 过拟合 |
| v2（lr 3.5e-5 / 10ep） | 0.9821（ep9 见顶） | ❌ 更差 |

温和优化未改善（v2 单尺度 0.9821 < v1 的 0.9865），ConvNeXt-Large@384 在该数据规模（57k 图）上无法超越 base@384（0.99147）。**容量上探方向彻底证伪**，建议停止 Large 投入。

## 6. 交付物

- 无有效交付物（未达门槛）。训练产物保留在云 `runs/cnxl_384_v2/`，权重不入库。
