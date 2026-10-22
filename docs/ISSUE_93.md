[Uploading ISSUE_93.md…]()
# Issue #93 结果记录 — Swin-B@384 16ep lr7e-5（lr 消融 vs lr1e-4）

> 对应 https://github.com/dongmingkun/glaucoma_first/issues/93 。与 seed42 基线（#76，lr 1e-4）唯一差异：lr 1e-4 → 7e-5、head 5e-4 → 3.5e-4。

## 训练（16 epoch，本机 RTX 5060 Laptop）

### 配置

| 项 | 值 |
|---|---|
| 命令 | `train.py --arch swin_base --img-size 384 --cache-root cache512 --epochs 16 --batch-size 4 --accum 12 --lr 7e-5 --head-lr 3.5e-4 --loader workers --workers 3 --seed 42 --tag swinb_384_e16_lr7e5` |
| 适配说明 | batch 8→4 / accum 6→12（8GB 显存），等效 batch 48 不变；与 #76 唯一差异 lr/head-lr |
| 环境 | Python 3.13.14 + torch 2.10.0+cu128 + RTX 5060 Laptop 8GB + cache512 |
| 耗时 | 09:07 → 17:38 ≈ 8.5h（前 41min/epoch 均衡电源、后 29min/epoch 野兽模式） |

### 每 epoch 曲线（EMA_AUROC / EMA_AUPRC）

```
ep1  0.96950 0.49109   ← 起点比 lr1e-4 的 0.95917 高（小 lr 无大震荡）
ep2  0.89027 0.22046   ← 震荡（幅度小于 lr1e-4 同期）
ep3  0.97773 0.65382
ep4  0.98316 0.75324
ep5  0.97373 0.72324
ep6  0.97585 0.73176
ep7  0.98059 0.76312
ep8  0.98106 0.77048
ep9  0.98223 0.77668
ep10 0.98509 0.78965  ← EMA 峰值
ep11 0.98211 0.77000
ep12 0.98155 0.76646
ep13 0.98238 0.77713
ep14 0.98237 0.77549
ep15 0.98269 0.77881
ep16 0.98293 0.78083   ← 收官仍小幅上探
```

## 验收结果

| 交付 | 结果 |
|---|---|
| valid TTA4（EMA 口径） | **AUROC 0.98535 / AUPRC 0.80305**（5849 行） |
| test 概率 CSV | `swinb_384_e16_lr7e5_test_probs_tta4.csv`，5855 行、无 NaN，已挂 Release |
| 权重 | best.pt（ep16，按本模型 auroc 选）+ last.pt，`runs/swin_base_384_1022_090749_swinb_384_e16_lr7e5/` |

## 工单核心答案：lr 7e-5 峰值更高，但终值打平

| 口径 | #76/#92 同族 (lr 1e-4) | 本 run (lr 7e-5) | 差 |
|---|---|---|---|
| EMA 曲线峰值 | 0.98465（seed123 ep9）/ 0.9917（队长 seed42 ep11） | **0.98509（ep10）** | seed123 对比 +0.0004 |
| valid TTA4 | 0.98599（seed123 本机） | 0.98535（本机） | -0.0006（噪声内） |
| 队长 seed42 本机参考 | **0.99233** | — | 本机复跑两 seed 均显著低于此值 |

**结论**：
1. 方向上验证"1e-4 偏大压峰值"（EMA 峰值 +0.0004），但幅度在单 seed 噪声边缘，不构成强证据；
2. valid TTA4 终值与 lr1e-4 打平（-0.0006），**未达 0.990 门槛**；
3. ⚠️ 重要观察：本机 5060 复跑的两个 seed（0.98599 / 0.98535）都显著低于队长 seed42 的 0.99233——除 seed 方差外，可能存在环境差异（电源模式降频 / 温度 / 版本），解读 3-Swin 平均时建议以队长本机数据为准。

**未跑阶段2**（工单禁止）。
