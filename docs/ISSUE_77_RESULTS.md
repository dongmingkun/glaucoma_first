# Issue #77 结果记录 — ViT-B@384 阶段1（8ep，train-only）

> 对应 https://github.com/dongmingkun/glaucoma_first/issues/77 。原文见 `docs/ISSUE_77.md`，本文件记录实测结果（16ep 配方更新后的结果跑完追加在文末）。

## 阶段1 训练（8 epoch，本机 RTX 5060 Laptop）

### 配置

| 项 | 值 |
|---|---|
| 命令 | `train.py --arch vit_base --img-size 384 --cache-root cache512 --epochs 8 --batch-size 4 --accum 12 --lr 5e-5 --head-lr 2.5e-4 --warmup-frac 0.10 --loader workers --workers 3 --seed 42 --tag vitb_384` |
| 适配说明 | batch 8→4 / accum 6→12（8GB 显存限制），等效 batch 48 不变 |
| 环境 | Python 3.13.14 + torch 2.10.0+cu128 + RTX 5060 Laptop 8GB，cache512（本地重建，63100 张） |
| 耗时 | 22:16 → 01:03 ≈ 2.8h（含一次 DataLoader 死锁 25 分钟后自恢复） |

### 每 epoch 曲线（train_loss / val_loss / AUROC / EMA_AUROC）

```
1: 0.465/0.386/0.8231/0.9393
2: 0.329/0.362/0.8597/0.9401
3: 0.275/0.416/0.8250/0.9603
4: 0.245/0.342/0.8979/0.9632
5: 0.223/0.248/0.9458/0.9624
6: 0.192/0.205/0.9586/0.9657  ← best.pt（EMA）
7: 0.174/0.315/0.9307/0.9605
8: 0.164/0.225/0.9566/0.9567
```

epoch 7/8 回落，best.pt 正确保留 epoch 6 权重（未被覆盖）。

### 验收结果

| 交付 | 结果 |
|---|---|
| valid TTA4 | **AUROC 0.96919 / AUPRC 0.67918**（best.pt = epoch 6 EMA，TTA4 比裸测 +0.0035） |
| test 概率 CSV | `vitb_384_test_probs_tta4.csv`，5855 行、三列、无 NaN、概率 ∈[0.0007, 0.9025]，已挂 Release `vitb_384-1021` |
| 权重 | best.pt / last.pt（690MB each），在 `runs/vit_base_384_1020_221628_vitb_384/` |

### 结论

未达 0.990 单模门槛（Swin-B 0.99233 仍是最强异构成员）。按工单约定交付 test 概率，供队长在干净 valid 上离线计算与现有集成的真实增益。

## 16ep 配方更新（队长指令，跑完追加）

- 依据 #76 Swin：8ep 0.98674 → 16ep 0.99233（+0.0046），8ep 明显欠训练。
- ViT-B 改跑 16ep，其余参数不变；若 8ep 时 valid TTA4 <0.985 可提前停。
- 状态：**进行中**。
