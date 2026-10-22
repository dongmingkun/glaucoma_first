# 复赛提交包说明

> **T4 状态（重要）**：当前**没有 T4 环境，100 ms 红线的 T4 p50/p95 均未实测**。
> 本仓库所有包在 5060 上的延迟都只是**下界参考**，**不能据此宣称通过 100 ms 红线**。
> 拿到 T4 后须按各包 README 的 `measure_latency.py` 命令真机复测并回填。

## 交付策略（issue #110：无 T4 方案）

官方红线：T4 单图端到端 ≤100 ms，但无 T4 可认证。改用「保守配置 + 安全边际」交付：

| 交付项 | 配置 | 5060 p95 目标 | 包体 | 状态 |
|---|---|---|---|---|
| **默认 · 单模型** | `submission_pkg_1model/`（cnxb_384_all_pseudo） | ≤40 ms（2.5× 余量） | ≤200 MB | 新建 |
| **默认 · 双模型** | `submission_pkg_2model/`（cb384 + cnxb, blend 0.65/0.35） | ≤50 ms（2× 余量） | ≤400 MB | 新建 |
| 候选 · 3 模型 | `submit_pkg/`（cb384 + cnxb + cs_hi512） | p95 ~80 ms | 452 MB | 仅作「有 T4 时」候选 |

- 默认交付 **单模型 / 双模型**（见下两目录的 README + `measure_latency.py`）。
- 3 模型包**仅作为有 T4 时的候选**，不作为默认交付（其 5060 p95 已近红线，T4 上风险最高）。
- **T4 栏一律写「未实测」**，不能写「已验证」。若有免费 T4（Kaggle/Colab/队友），10 分钟补测即可，不作为阻塞项。

## 默认包目录

- `submission_pkg_1model/` —— 单模型（cnxb_384_all_pseudo），命令/T4 策略/组装见其 README。
- `submission_pkg_2model/` —— 双模型（cb384 + cnxb，blend 0.65/0.35），同上。

两个包均自带 `predict.py`（与下面主版逐行一致）+ `measure_latency.py` + `assemble.py` + `MANIFEST.md`。

---

## 主推理脚本（submit_pkg/predict.py，3 模型包）

### 内容（权重合计 452.4MB ≤ 500MB）

| 文件 | 大小 | 说明 |
|---|---|---|
| `predict.py` | 14KB | 自包含推理脚本（多线程预处理 + 有界预取 + 多模型加权） |
| `cb384.pt` | 176.3MB | ConvNeXt-Base@384，train-only，valid TTA4 AUROC 0.99147，cache512 |
| `cnxb_384_all_pseudo.pt` | 176.3MB | ConvNeXt-Base@384，train+all + 测试域伪标签，check PASS 0.9022，cache512 |
| `cs_hi512_all_ra.pt` | 99.8MB | ConvNeXt-Small@512，train+all + RandAugment，check PASS 0.9132，cache512 |

三个成员同一 cache 档位（512），每张图只解码/裁剪一次。

### 推荐命令（3 模型 + TTA1，仅 T4 候选）

```bash
python predict.py \
  --images /path/to/test_images \
  --csv /path/to/test.csv \
  --weights cb384.pt cnxb_384_all_pseudo.pt cs_hi512_all_ra.pt \
  --blend 0.5 0.25 0.25 \
  --tta 1 --batch-size 32 --prep-workers 8 \
  --out submission.csv
```

- **T4 未实测**（当前无 T4 环境）：下面的 70–90 ms 只是估算，不能据此宣称 100ms 红线通过；实测记录见 `docs/ISSUE_81.md`。
- 本机 p95 受共享 GPU / 70W 功耗墙影响；2 模型对照复测曾出现 p95 120ms（系统抖动，非模型数导致），因此**降级配置不保证更快，必须以 T4 实测为准**。
- 预处理已优化：JPEG draft（512 缓存 → 768）+ 圆形视野 bbox + 居中方形 + BICUBIC；
  `--prep-workers 8` 实测把解码/裁剪从 64 → 21 ms/图（1→8 线程），并与 GPU 前向重叠。

### 降级配置（2 模型，权重 352.6MB，已封装为 `submission_pkg_2model/`）

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

## 已验证（2026-10-21，详见 `docs/ISSUE_81.md`）

- **官方版本环境跑通**：Python 3.12.10 + torch 2.10.0+cu128 + torchvision 0.25.0+cu128（CUDA 12.8 / cuDNN 91002）；
- **输出一致性**：前 5 行与参考逐位一致；与 Python 3.11 参考（300 张）最大差 **1.21e-4**；同环境连续两次全量运行逐位相同；
- **本地延迟参考**：RTX 5060 Laptop，3 模型 TTA1，全量 5855 张两轮 **304s / 338s（52–58 ms/图）**，稳态 p50 **37.5–45.5 ms**、p95 **77.2–80.7 ms**；
- **包体**：包内实际文件合计 **452.48 MB ≤ 500MB**；
- **T4 未实测**：当前无 T4 环境；100ms 红线未在 T4 认证（见 `docs/ISSUE_81.md`）。

## 合规

- 全部权重为 ConvNeXt，仅用主办方数据训练；**不含 RetFound 或任何外部医学预训练权重**。
- 无外部标注数据、无外部标签。
