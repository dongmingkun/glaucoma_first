# Issue #76 — 异构架构多样性：Swin-B@384（阶段1 达标，待队长确认阶段2）

> 状态：阶段1 完成，valid TTA4 达标（0.99233 ≥ 0.990）｜ 指派人：Yulingling-sc ｜ 执行环境：AutoDL 4090D
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
