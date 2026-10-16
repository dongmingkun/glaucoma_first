# Issue #8 — 建模探索池（协议 + 代码进展）

> 逐项在 valid 上验证，**有稳定增益才纳入集成**；失败也回填本 Issue（负结果同样有价值）。
> 判据不变：**AUROC +0.001 或 AUPRC +0.005** 才算「值得纳入集成」。

## 0. 基线（对照用，768 缓存 / 高分辨率时代）

- 单模 Small@512（768 缓存）TTA4 = **AUROC 0.98994 / AUPRC 0.87256**
- 最佳集成 = **AUROC 0.99110 / AUPRC 0.87733**

⚠️ 本池实验请在 **768 缓存 / 高分辨率配方**上做（512 缓存时代的结论可能不成立）。

## 1. 候选清单（按性价比）

### 1.1 ⭐ 多尺度 TTA（代码已就绪，最便宜，优先做）

本 PR 已给 `evaluate.py` / `infer.py` 加 `--img-sizes`（同模型多尺度推理取平均）：

```bash
# 单尺度（原有行为）
python evaluate.py --ckpt <hi512/best.pt> --img-size 512 --tta 4

# 多尺度 TTA（3 尺度平均，0.5 小时出结论）
python evaluate.py --ckpt <hi512/best.pt> --img-sizes 448 512 576 --tta 4
```

原理：同一模型在不同输入尺度下推理，概率平均——与 flip TTA 正交，几乎零成本
（复用 cache512 图片直接 resize）。预期对 AUPRC 有小幅正效应（类似分辨率集成的收益）。

### 1.2 mixup / cutmix（代码待做）

在已收敛模型上 +2~3 epoch 微调。需在 `train.py` 增广处加 mixup/cutmix 开关
（`--mixup <alpha>` / `--cutmix <alpha>`），建议 alpha 0.2 起步。

```bash
python train.py --arch convnext_tiny --img-size 512 --epochs 3 \
  --init-from <best.pt> --mixup 0.2 --tag mixup_probe
```

### 1.3 损失 / pos_weight 敏感度（代码待做）

`pos_weight ∈ {3, 8, 12}` 与 focal loss 各跑 1~2 epoch 对比。需在 `train.py` 加
`--focal-gamma`（focal loss，gamma 2.0 起步，AUPRC 可能有正效应）。

```bash
python train.py --arch convnext_tiny --img-size 512 --epochs 2 --init-from <best.pt> \
  --pos-weight 8 --focal-gamma 2.0 --tag focal_probe
```

### 1.4 年龄融合升级 → ❌ 已剔除

已有消融：无年龄 0.98434 vs 有年龄 0.98492，增量 ~0.0006，不值得投入。

### 1.5 增广增强（RandAugment / 更大旋转）— 次优先，代码待做

`train.py` 加 RandAugment（`--randaugment N M`）或把旋转范围从 ±25° 提到 ±30°。

### 1.6 Seed ensemble（最稳最贵，放最后）

`train.py` 已有 `--seed`，同配置跑 2~3 个 seed 再 `ensemble.py` 平均。等高分辨率成员到位后再做。

## 2. 结论表（逐项回填，成功/失败都记）

| 候选 | 命令摘要 | epoch | val AUROC | val AUPRC | Δ vs 基线 | 判据 | 结论 |
|---|---|---|---|---|---|---|---|
| 多尺度 TTA | `--img-sizes 448 512 576` | - | 待回填 | 待回填 | 待回填 | ≥+0.001/+0.005 | 待回填 |
| mixup/cutmix | `--mixup 0.2` | 待回填 | 待回填 | 待回填 | 待回填 | 同上 | 待回填 |
| focal loss | `--focal-gamma 2.0` | 待回填 | 待回填 | 待回填 | 待回填 | 同上 | 待回填 |
| RandAugment | `--randaugment` | 待回填 | 待回填 | 待回填 | 待回填 | 同上 | 待回填 |
| Seed ensemble | 2~3 seed | 待回填 | 待回填 | 待回填 | 待回填 | 同上 | 待回填 |

## 3. 交付物状态

- [x] 多尺度 TTA 代码（evaluate.py / infer.py `--img-sizes`）
- [ ] mixup / focal / RandAugment 代码（train.py，待确认优先级后补）
- [ ] 每项实验结论表（需 GPU 机器实测回填）
- [ ] 值得纳入集成的候选列表 + 权重路径

## 4. Issue 回填模板（每项跑完贴一条）

```
[探索池] 候选=<名> ckpt=<run> 命令=<...> epoch=<n> 
AUROC=<x> AUPRC=<y> balACC@0.5=<z> Δ vs 基线=<±d> 耗时=<t>
结论=<纳入集成 / 无增益 / 负向>
```
