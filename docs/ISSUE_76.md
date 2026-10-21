# Issue #76 — 异构架构多样性：Swin-B@384（阶段1 达标，待队长确认阶段2）

> 状态：阶段1+阶段2 完成（阶段2 train-on-all 验收 PASS，Spearman 0.8753→0.869→0.9066）｜ 指派人：Yulingling-sc ｜ 执行环境：AutoDL 4090D
> 本文为 issue #76 阶段1 结果的归档；**权威内容以 GitHub issue #76 评论为准**。

---

## 1. 背景

权重线已关闭（峰值 20% 混合 98.82），剩余提分只能靠**多样性**。现有成员全是 ConvNeXt（+少量 EfficientNet），本工单引入第一个真正异构的强骨干 Swin-B（window attention）。

队长本地先验：Swin-B 8ep valid TTA4 0.98774（多样性极好、与 ens9 秩相关 0.7536），但质量不够 w=0。工单调整：16ep 单变量验证「是否只是欠训练」。

## 2. 训练配置（从零 train-only 16ep）

```bash
python train.py --arch swin_base --img-size 384 --cache-root cache512 \
  --epochs 16 --batch-size 8 --accum 6 --lr 1e-4 --head-lr 5e-4 \
  --loader workers --workers 3 --seed 42 --tag swinb_384_e16
```

| 项 | 值 |
|---|---|
| 架构 / 分辨率 | Swin-B @384（timm swin_base_patch4_window12_384，88M，window attention） |
| 训练方式 | 从零训练（ImageNet 原生 384 权重初始化） |
| epoch / seed | 16 / 42 |
| 显存 / 耗时 | 6.3G / 每 epoch ≈ 12.7 min，16ep 约 3.4h |

（前置：train.py timm 异构骨干支持 commit 27178b7；云上 HF 直连不通，用 `HF_ENDPOINT=https://hf-mirror.com` 镜像下载 swin_base 权重。）

## 3. 训练曲线（EMA AUROC）

| epoch | EMA AUROC | epoch | EMA AUROC |
|---|---|---|---|
| 1 | 0.9733 | 9 | 0.9901 |
| 2 | 0.9779 | 10 | 0.9912 |
| 3 | 0.9815 | **11** | **0.9917（峰值）** |
| 4 | 0.9843 | 12 | 0.9910 |
| 5 | 0.9854 | 13 | 0.9898 |
| 6 | 0.9870 | 14 | 0.9896 |
| 7 | 0.9884 | 15 | 0.9891 |
| 8 | 0.9895 | 16 | 0.9887 |

## 4. 验收

`infer.py --ckpt runs/swinb_384_e16/best.pt --split valid --tta 4`（use=ema）：

- **valid AUROC = 0.99233 / AUPRC = 0.85930** ≥ 0.990 门槛 ✅
- **超过 base@384（0.99147）约 +0.0009**

对比 8ep（0.98774）→ 16ep（0.99233），**「欠训练」判断成立**。

## 5. 交付物（Release `weights-1016`）

| 文件 | 说明 |
|---|---|
| `swinb_384_test_probs_tta4.csv` | test TTA4 概率，5855 行，未校准 |

## 6. 结论与下一步

- 阶段1 达标，Swin-B@384 16ep 是首个 valid 超过 base@384 的异构成员，且多样性好（秩相关 ~0.75）。
- **待队长离线算它与现有集成的真实增益**；若为正，触发阶段2（train+all + 伪标签，一次连续跑完）。

---

## 7. 阶段2：train-on-all（全量重训，三次尝试后验收 PASS）

> 触发条件：队长确认阶段1 与现有集成的增益为正（重建 ens9 AUROC 0.99260，秩相关 0.7693）→ 触发阶段2。

### 7.1 第一次尝试（REJECT）

首跑 `--epochs 8 --extra-rows pseudo/...`（train-on-all + 伪标签），`check_train_on_all.py` 验收：

- Spearman **0.8753 < 0.90** ❌ → REJECT
- 归因：epoch 数偏少（阶段1 最佳 epoch=11，阶段2 应 ≈ round(11/1.114)=10）+ 叠加伪标签分布漂移

### 7.2 第二次尝试（REJECT）

干净 train-on-all（折入 valid，不加伪标签），`--epochs 10`：

- Spearman **0.869 < 0.90** ❌ → REJECT（比首跑更低）
- 诊断：减 epoch 反而更差，确认是**真实排序漂移**而非伪标签导致；且候选 `auroc/auroc_ema` 均 = -1（valid 已污染），`use` 自动判定走 **ema**，与参考一致——故 EMA 选择不是问题，根因是 train-on-all 在 test 上与 train-only 参考的排序一致性不足。

### 7.3 第三次尝试（PASS）

降 LR + 减 epoch（降低从 warm-start 的发散量，保留 train-on-all 收益）：

```bash
python train.py --arch swin_base --img-size 384 --cache-root cache512 \
  --epochs 5 --batch-size 8 --accum 6 --lr 5e-5 --head-lr 2.5e-4 \
  --loader workers --workers 3 --seed 42 --train-on-all \
  --init-from runs/swinb_384_e16/best.pt --tag swinb_384_all_v3
```

`check_train_on_all.py --ckpt .../swinb_384_all_v3/best.pt --ref runs/swinb_384_e16/best.pt --tta 4`（use=ema）：

- **Spearman = 0.9066 ≥ 0.90** ✅（排序未崩）
- mean_prob 相对偏差 = 0.063 ≤ 0.25 ✅（尺度未漂移）
- top200 重叠率 = 0.945
- 阶段2 mean_prob = 0.05561 vs 阶段1参考 0.05934

### 7.4 交付物（Release `weights-1016`）

| 文件 | 说明 |
|---|---|
| `swinb_384_all_v3_test_probs_tta4.csv` | 阶段2 test TTA4 概率，5855 行，未校准（集成用） |
| `swinb_384_all_v3_fp16.pt` | 阶段2 fp16 瘦身推理权重（集成成员，体积预算内） |

### 7.5 结论

- 阶段2 train-on-all 经三次迭代后通过验收，Swin-B@384 成为可集成的异构成员候选（fp16 权重可直接入包）。
- 真值增益由队长离线并入集成后判定。
