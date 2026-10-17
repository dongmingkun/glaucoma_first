# Issue #8 — 建模探索池（协议 + 代码进展 + 离线快照）

> 状态：open ｜ 指派人：Yulingling-sc
> **权威内容以 GitHub issue #8 为准**；本文件供队友的 AI 工具 / 离线环境直接读取（快照同步：2026-10-16）。
> 结构：Part A = 实验协议与代码进展（PR #26 交付）；Part B = issue 正文与全部评论的离线快照。

---

# Part A：实验协议与代码进展

> 逐项在 valid 上验证，**有稳定增益才纳入集成**；失败也回填本 Issue（负结果同样有价值）。
> 判据不变：**AUROC +0.001 或 AUPRC +0.005** 才算「值得纳入集成」。

## A0. 基线（对照用，768 缓存 / 高分辨率时代）

- 单模 Small@512（768 缓存）TTA4 = **AUROC 0.98994 / AUPRC 0.87256**
- 最佳集成 = **AUROC 0.99110 / AUPRC 0.87733**

⚠️ 本池实验请在 **768 缓存 / 高分辨率配方**上做（512 缓存时代的结论可能不成立）。

## A1. 候选清单（按性价比）

### A1.1 ⭐ 多尺度 TTA（代码已就绪，最便宜，优先做）

`evaluate.py` / `infer.py` 已加 `--img-sizes`（同模型多尺度推理取平均）：

```bash
# 单尺度（原有行为，输出文件名与旧版一致）
python evaluate.py --ckpt <hi512/best.pt> --img-size 512 --tta 4

# 多尺度 TTA（3 尺度平均，0.5 小时出结论）
python evaluate.py --ckpt <hi512/best.pt> --img-sizes 448 512 576 --tta 4
```

原理：同一模型在不同输入尺度下推理，概率平均——与 flip TTA 正交，几乎零成本
（复用 cache512 图片直接 resize）。预期对 AUPRC 有小幅正效应（类似分辨率集成的收益）。

> 产物命名：单尺度仍为 `probs_valid_full_tta4.npy`（保持原有约定，`compare_ensembles.py` 等依赖此名）；
> 多尺度为 `probs_valid_full_ms448-512-576_tta4.npy`。

### A1.2 mixup / cutmix（代码待做）

在已收敛模型上 +2~3 epoch 微调。需在 `train.py` 增广处加 mixup/cutmix 开关
（`--mixup <alpha>` / `--cutmix <alpha>`），建议 alpha 0.2 起步。

```bash
python train.py --arch convnext_tiny --img-size 512 --epochs 3 \
  --init-from <best.pt> --mixup 0.2 --tag mixup_probe
```

### A1.3 损失 / pos_weight 敏感度（代码待做）

`pos_weight ∈ {3, 8, 12}` 与 focal loss 各跑 1~2 epoch 对比。需在 `train.py` 加
`--focal-gamma`（focal loss，gamma 2.0 起步，AUPRC 可能有正效应）。

```bash
python train.py --arch convnext_tiny --img-size 512 --epochs 2 --init-from <best.pt> \
  --pos-weight 8 --focal-gamma 2.0 --tag focal_probe
```

### A1.4 年龄融合升级 → ❌ 已剔除

已有消融：无年龄 0.98434 vs 有年龄 0.98492，增量 ~0.0006，不值得投入。

### A1.5 增广增强（RandAugment / 更大旋转）— 次优先，代码待做

`train.py` 加 RandAugment（`--randaugment N M`）或把旋转范围从 ±25° 提到 ±30°。

### A1.6 Seed ensemble（最稳最贵，放最后）

`train.py` 已有 `--seed`，同配置跑 2~3 个 seed 再 `ensemble.py` 平均。等高分辨率成员到位后再做。

## A2. 结论表（逐项回填，成功/失败都记）

| 候选 | 命令摘要 | epoch | val AUROC | val AUPRC | Δ vs 基线 | 判据 | 结论 |
|---|---|---|---|---|---|---|---|
| 多尺度 TTA | `--img-sizes 448 512 576` | - | **0.99070** | **0.87796** | AUROC +0.00076 / AUPRC **+0.00540** | ≥+0.001/+0.005 | ✅ 值得纳入（AUPRC 过 +0.005 线，边际） |
| mixup/cutmix | `--mixup 0.2` | 3ep 探针 / 6ep 复验 | 探针 0.98873 / **复验 0.99006** | 探针 0.85791 / 复验 0.86974 | 探针 +0.00385 / **复验 −0.00064** | AUROC≥+0.001 | ⚠️ 探针有效但主力配方复验负（0.99006 < 基线 0.99070），不纳入 |
| focal loss | `--focal-gamma 2.0` | 3ep 热启动 | **0.98918** | **0.84375** | AUROC +0.00430 / AUPRC +0.00617 | AUROC≥+0.001 | ✅ 有效（边际，主力复验未做） |
| RandAugment | `--randaugment 2 9` | 3ep 探针 / 6ep 复验 | 探针 0.98992 / **复验 0.99188** | 探针 0.85677 / 复验 0.87261 | 探针 +0.00504 / **复验 +0.00118** | AUROC≥+0.001 | ✅ 达标（唯一有效，多 seed 复验产出成员中） |
| Seed ensemble | Base@384 seed7 从头训 | 6ep | **0.98801** | **0.85441** | −0.00346 vs 0.99147 | ≥追平 0.99147 | ❌ 负（seed 多样性无正收益） |

> ⚠️ **判据更新（2026-10-17）**：评分口径已锁定为 **AUROC×100**（队长提交 ens8c 实测 98.15 反推），探针判据改为 **AUROC ≥ +0.001**（此前 AUPRC 判据作废）。上表 3 项训练侧探针（mixup/focal/RandAugment）均为 TTA1 val ep3 EMA，vs baseline（AUROC 0.98488 / AUPRC 0.83758，同配置同 seed 42 同 3ep 热启动）。

## A3. 交付物状态

- [x] 多尺度 TTA 代码（evaluate.py / infer.py `--img-sizes`）
- [x] mixup / focal / RandAugment 代码（train.py 开关，PR #40）
- [x] 多尺度 TTA 结论表（#30，2026-10-16 实测回填）
- [x] mixup / focal / RandAugment 探针（#45，2026-10-17 云 4090D 实测回填）
- [x] mixup / RandAugment / Seed ensemble 主力配方复验（#51/#52，2026-10-17：mixup 负、Base@384 seed7 负、ra 重跑中）

## A4. Issue 回填模板（每项跑完贴一条）

```
[探索池] 候选=<名> ckpt=<run> 命令=<...> epoch=<n> 
AUROC=<x> AUPRC=<y> balACC@0.5=<z> Δ vs 基线=<±d> 耗时=<t>
结论=<纳入集成 / 无增益 / 负向>
```

---

# Part B：离线快照（issue 正文 + 全部评论）

## B1. 正文

## 背景（为什么要做）

基础管线已收敛到 valid AUROC 0.985 / AUPRC 0.816。下面是候选提升方向，按性价比排序，**逐项在 valid 上验证，有稳定增益才纳入最终集成**；验证失败的也回填本 Issue（负结果同样有价值，避免重复踩）。

## 候选清单（按建议顺序）

1. **多尺度 TTA**：384 模型在 384 + 448 两种尺度分别推理后平均（给 `evaluate.py`/`infer.py` 加尺度参数，或写临时脚本）。
2. **mixup / cutmix**：在已收敛模型上 + 2~3 epoch 微调，看 AUROC/AUPRC 变化。
3. **损失 / 权重敏感度**：pos_weight ∈ {3, 8, 12}、focal loss，各跑 1~2 epoch 对比。
4. **年龄融合升级**：age 分桶 embedding，或把年龄放到更晚的融合位置（已有消融：无年龄 0.98434 vs 有年龄 0.98492，增量很小）。
5. **增广增强**：RandAugment、更大旋转范围（±30°）、更强颜色抖动。
6. **Seed ensemble**：同配置 2~3 个随机种子训练后平均（最稳但最贵，放最后）。

## 协议（每项都要遵守）

- 记录：命令、epoch 数、val AUROC/AUPRC/平衡准确率@0.5、耗时
- 判据：AUROC 提升 ≥0.001 或 AUPRC 提升 ≥0.005 才算"值得纳入集成"
- 产物命名规范见 Issue #6；结论回填本 Issue 评论

## 交付物

- [ ] 每项实验一行结论表（成功/失败都记录）
- [ ] 值得纳入集成的候选列表 + 对应权重路径

## B2. 评论

### @dongmingkun (2026-10-14 15:12)

**本工单负责人：@Yulingling-sc**（训练方向）。接受 GitHub 仓库协作邀请后即可查看/操作本仓库；如需调整分工，请在本工单回复。

### @dongmingkun (2026-10-15 15:01)

## 上下文刷新（2026-10-15，基线已大幅前移）

本池写于"AUROC 0.985 / AUPRC 0.816"时代，现已达到 **AUPRC 0.877（集成）**。逐项重新评估：

| 原候选 | 现状 | 建议 |
|---|---|---|
| 1. 多尺度 TTA | **仍未验证，且最便宜** | ⭐ 优先做：对 hi512 模型在 448/512/576 三尺度推理取平均，0.5 小时出结论 |
| 2. mixup / cutmix | 仍未验证 | 值得试（在 hi512 已收敛权重上 +2~3 epoch） |
| 3. 损失/pos_weight 敏感度 | 仍未验证 | 值得试（尤其 focal loss 对 AUPRC 可能有正效应） |
| 4. 年龄融合升级 | **已有结论**：无年龄 0.98434 vs 有年龄 0.98492，增量 ~0.0006，不值得投入 | ❌ 剔除 |
| 5. 增广增强（RandAugment/更大旋转） | 仍未验证 | 次优先 |
| 6. Seed ensemble | 仍未验证 | 稳但贵；在高分辨率成员到位后再做 |

**新基线（对照用）**：Small@512（768缓存）TTA4 = AUROC 0.98994 / AUPRC 0.87256；最佳集成 = 0.99110 / 0.87733。
**判据不变**：AUROC +0.001 或 AUPRC +0.005 才纳入集成；失败也回填本 issue。

⚠️ 注意：本池的实验请**在 768 缓存/高分辨率配方上做**（512 缓存时代的结论可能不成立）。