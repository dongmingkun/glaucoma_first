# 复赛提交包说明（2026-10-20 更新）

## 内容（权重合计 452.4MB ≤ 500MB）

| 文件 | 大小 | 说明 |
|---|---|---|
| `predict.py` | 14KB | 自包含推理脚本（多线程预处理 + 有界预取 + 多模型加权） |
| `cb384.pt` | 176.3MB | ConvNeXt-Base@384，train-only，valid TTA4 AUROC 0.99147，cache512 |
| `cnxb_384_all_pseudo.pt` | 176.3MB | ConvNeXt-Base@384，train+all + 测试域伪标签，check PASS 0.9022，cache512 |
| `cs_hi512_all_ra.pt` | 99.8MB | ConvNeXt-Small@512，train+all + RandAugment，check PASS 0.9132，cache512 |

三个成员同一 cache 档位（512），每张图只解码/裁剪一次。

## 推荐命令（3 模型 + TTA1）

```bash
python predict.py \
  --images /path/to/test_images \
  --csv /path/to/test.csv \
  --weights cb384.pt cnxb_384_all_pseudo.pt cs_hi512_all_ra.pt \
  --blend 0.5 0.25 0.25 \
  --tta 1 --batch-size 32 --prep-workers 8 \
  --out submission.csv
```

- **实测延迟**（RTX 5060 Laptop，TTA1，3 模型）：**42.9 ms/图**（5855 张 251s）。
- T4 估算：前向约慢 1.5–2× → 预计 **70–90 ms/图**；若实测 p95 超 100ms，用下面的降级配置。
- 预处理已优化：JPEG draft（512 缓存 → 768）+ 圆形视野 bbox + 居中方形 + BICUBIC；
  `--prep-workers 8` 实测把解码/裁剪从 64 → 21 ms/图（1→8 线程），并与 GPU 前向重叠。

## 降级配置（2 模型，权重 352.6MB）

```bash
python predict.py \
  --images /path/to/test_images \
  --csv /path/to/test.csv \
  --weights cb384.pt cnxb_384_all_pseudo.pt \
  --blend 0.65 0.35 --tta 1 --batch-size 32 --prep-workers 8 \
  --out submission.csv
```

## 与线上提交的关系（重要）

- 线上最佳 **98.82** 来自 9 成员 × TTA4 的集成，权重合计超出 500MB/100ms 约束；
  本包是**合规子集**（3 模型 TTA1），与 ens10 的排序相关约 0.94，预期分数略低。
- 若官方复赛只校验格式/可复现性而不重新打分，最终仍以线上最佳 CSV 为准；
  若官方用本包重跑隐藏测试集，则以本包分数为准。两种口径都保持本包合规。

## 输出格式

UTF-8 CSV，三列：`Image_name,Probability,Pred_Label`（Probability ∈ [0,1]，
`Pred_Label` 以 0.5 为阈值；复赛指标为 AUROC，阈值/校准不影响得分）。

## 设计说明

- **自包含**：脚本内部实现眼底裁剪，不依赖训练缓存；权重含 `arch/img_size/cache_size/use` 元信息。
- **选权**：尊重 checkpoint 的 `use` 字段（train+all 权重为 `ema`）。
- **多架构**：支持 `efficientnet_b0/b3/b4`、`convnext_tiny/small/base`、`resnet50`。
- **年龄**：缺失或异常（<1 或 >110）用训练中位数 58；正常值 clip 到 [10,100]。
- **多线程**：`--prep-workers`（默认 8）；PIL/numpy 释放 GIL，热核数超过 8 后收益趋平。
- **有界预取**：解码/裁剪与 GPU 前向流水线重叠，端到端 ≈ max(预处理, 前向) 而非两者之和。

## 合规

- 全部权重为 ConvNeXt，仅用主办方数据训练；**不含 RetFound 或任何外部医学预训练权重**。
- 无外部标注数据、无外部标签。
