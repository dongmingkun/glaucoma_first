[ISSUE_92.md](https://github.com/user-attachments/files/33818014/ISSUE_92.md)
# Issue #92 结果记录 — Swin-B@384 16ep seed123（3-Swin 平均成员）

> 对应 https://github.com/dongmingkun/glaucoma_first/issues/92 。本文件记录 seed123 版实测结果。

## 训练（16 epoch，本机 RTX 5060 Laptop）

### 配置

| 项 | 值 |
|---|---|
| 命令 | `train.py --arch swin_base --img-size 384 --cache-root cache512 --epochs 16 --batch-size 4 --accum 12 --lr 1e-4 --head-lr 5e-4 --loader workers --workers 3 --seed 123 --tag swinb_384_e16_s123` |
| 适配说明 | batch 8→4 / accum 6→12（8GB 显存），等效 batch 48 不变；与 seed42 基线唯一差异：seed |
| 环境 | Python 3.13.14 + torch 2.10.0+cu128 + RTX 5060 Laptop 8GB + cache512 |
| 耗时 | 19:42 → 07:01 ≈ 11.2h（~44min/epoch，Windows 均衡电源模式） |

### 每 epoch 曲线（EMA_AUROC / EMA_AUPRC）

```
ep1  0.95917 0.38316
ep2  0.96761 0.65408
ep3  0.97662 0.66400
ep4  0.97906 0.73931
ep5  0.98083 0.76544
ep6  0.98161 0.77900
ep7  0.98399 0.80251
ep8  0.98422 0.81202
ep9  0.98465 0.80359  ← EMA 峰值
ep10 0.98347 0.80372
ep11 0.98320 0.79413
ep12 0.98247 0.79177
ep13 0.98189 0.78273
ep14 0.98227 0.78094
ep15 0.98211 0.77545
ep16 0.98244 0.77771
```

曲线形态与 seed42 基线（#76，ep11 见顶 0.9917 后缓降）同型；本 run ep9 见顶 0.98465 后缓降。

## 验收结果

| 交付 | 结果 |
|---|---|
| valid TTA4（EMA 口径） | **AUROC 0.98599 / AUPRC 0.80995**（5849 行） |
| test 概率 CSV | `swinb_384_e16_s123_test_probs_tta4.csv`，5855 行、无 NaN、概率 ∈[0.000276, 0.999507]，已挂 Release |
| 权重 | best.pt / last.pt（700MB each），`runs/swin_base_384_1021_194217_swinb_384_e16_s123/` |

## 两个需要队长解读 3-Swin 平均时注意的点

1. **best.pt 选择口径**：train.py 的 best 按**本模型 auroc** 选（挑中 ep14，auroc 0.98829），而 EMA 峰值在 ep9（0.98465）——ep9 权重已被覆盖未保存。本交付的 EMA 口径是 ep14 的 EMA（0.98227），略低于 EMA 峰值。
2. **seed 方差**：同配方只换 seed，本 run 0.98599 vs 队长 seed42 的 0.99233，差 0.0063——接近或超过 3-Swin 平均期望的方差缩减收益，建议队长结合 ens9 相关性综合解读。

## 结论

未达 0.990 单模门槛（与 seed42 基线同型），按工单约定交付 test 概率，供离线计算 3-Swin 平均对 ens9 的干净 valid 增益。**未跑阶段2**（工单禁止）。
