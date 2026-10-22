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
| valid TTA4（EMA 口径） | **AUROC 0.985995 / AUPRC 0.80995**（5849 行，已核验；Release [`swinb_e16_s123-1022`](https://github.com/dongmingkun/glaucoma_first/releases/tag/swinb_e16_s123-1022)） |
| test 概率 CSV | `swinb_384_e16_s123_test_probs_tta4.csv`，5855 行、无 NaN、概率 ∈[0.000276, 0.999507]，Release [`swinb_e16_s123-1022`](https://github.com/dongmingkun/glaucoma_first/releases/tag/swinb_e16_s123-1022)（已核验行数/顺序/范围） |
| 权重 | best.pt / last.pt（700MB each），`runs/swin_base_384_1021_194217_swinb_384_e16_s123/` |

## 交付物核验（2026-10-22）

- ✅ valid CSV：5849 行、`Image_name` 顺序与 `valid.csv` 一致、无 NaN；复算 **AUROC 0.985995 / AUPRC 0.80995**，与文档一致。
- ✅ test CSV：5855 行、顺序与 `test.csv` 一致、无 NaN，范围与文档一致。
- 文件已由队长补挂到 Release `swinb_e16_s123-1022`。

## 3-Swin 平均增益（队长侧计算，2026-10-22）

| 成员 | valid AUROC | valid AUPRC |
|---|---|---|
| seed42（#76，云端） | 0.992331 | 0.85930 |
| seed7（#90） | 0.991892 | 0.85676 |
| seed123（本单） | 0.985995 | 0.80995 |
| **3-Swin 等权平均（概率）** | **0.992197** | **0.86690** |
| 3-Swin 等权平均（logit） | 0.992248 | — |
| ens9 基线 | 0.992604 | — |

- 秩相关：s42-s7 **0.8737**、s42-s123 **0.8771**、s7-s123 **0.8842**；3-Swin 平均 vs ens9 **0.8069**。
- 与 ens9 混合（概率平均）：最优在 w≈0.48，AUROC **0.992890**（**+0.000286**）；配对 bootstrap **P(Δ>0)=0.842**。
- logit 平均 w=0.30 时 +0.000316，但同样未通过 P≥0.9 门槛。
- **判定：无显著增益，不进集成。** 3-Swin 平均本身（0.992197）低于 seed42 单模（0.992331），因为 seed123 的 0.985995 明显拖低了平均；seed 方差远大于平均带来的方差缩减收益。

## 两个需要队长解读 3-Swin 平均时注意的点

1. **best.pt 选择口径**：train.py 的 best 按**本模型 auroc** 选（挑中 ep14，auroc 0.98829），而 EMA 峰值在 ep9（0.98465）——ep9 权重已被覆盖未保存。本交付的 EMA 口径是 ep14 的 EMA（0.98227），略低于 EMA 峰值。
2. **seed 方差**：同配方只换 seed，本 run 0.98599 vs 队长 seed42 的 0.99233，差 0.0063——接近或超过 3-Swin 平均期望的方差缩减收益，建议队长结合 ens9 相关性综合解读。

## 结论

未达 0.990 单模门槛；3-Swin 平均对 ens9 的增益 +0.000286（P=0.842），低于团队 P≥0.9 门槛，**不进集成**。本单按负结果关闭；**未跑阶段2**（工单禁止）。
