# 降级提交包（单模型）— submission_pkg_1model

> 对应工单 **#110**：无 T4 方案。本包是「保守配置 + 安全边际」交付的**默认单模型**选项。
> **T4 未实测**（当前无 T4 环境）；下方延迟数字均为 RTX 5060 实测/估算，仅作下界参考。

## 内容（单模型，包体 ≤200 MB）

| 文件 | 大小 | 说明 |
|---|---|---|
| `predict.py` | 14 KB | 自包含推理脚本（与 `submit_pkg/predict.py` 主版本逐行一致） |
| `cnxb_384_all_pseudo.pt` | ~176 MB | ConvNeXt-Base@384，train+all + 测试域伪标签，cache512，fp16 单态 |
| `measure_latency.py` | 6 KB | 逐批计时工具（mean/p50/p95，剔除前 2 warmup 批） |
| `assemble.py` | 3 KB | 从 Release `weights-1016` 拉权重 + 瘦身 cnxb（无 torch 时给预瘦身地址） |
| `MANIFEST.md` | — | 权重 md5 / 元信息 / 包体校验记录 |

权重元信息（由 `MANIFEST.md` 实测）：`arch=convnext_base` / `img_size=384` /
`cache_size=512` / `use=ema`。

## 推荐命令（TTA1，单模型）

```bash
python predict.py \
  --images /path/to/test_images \
  --csv /path/to/test.csv \
  --weights cnxb_384_all_pseudo.pt \
  --tta 1 --batch-size 32 --prep-workers 8 \
  --out submission.csv
```

## 延迟与安全边际（5060，下界；**T4 未实测**）

- **5060 实测（2026-10-23，全量 5855 张，bs32，prep-workers 8）**：
  - mean **32.65 ms** / p50 **29.18 ms** / **p95 53.27 ms** / p99 75.58 ms（181 个计时 batch，剔除前 2 个 warmup）。
  - 目标 p95 ≤40 ms：**未达到（53.27 ms）**；相对 100 ms 红线约 **1.9×** 余量。
- 原因：5060 的 p95 受共享 GPU / 70W 功耗墙 / 系统抖动影响；p50 29 ms 是稳定值。**T4 未实测，100 ms 红线不能据此认证。**

```bash
python measure_latency.py \
  --images /path/to/test_images --csv /path/to/test.csv \
  --weights cnxb_384_all_pseudo.pt \
  --tta 1 --batch-size 32 --prep-workers 8
```

## 组装 / 校验

```bash
python assemble.py --mode 1model     # 拉权重 + 瘦身 cnxb（需 torch；否则给预瘦身地址）
```

- 包体：单模型目录 ≤200 MB ✅（cnxb_384_all_pseudo.pt ≈176 MB）。
- `predict.py` 与 `submit_pkg/predict.py` 主版本一致，仅复制、未改推理逻辑。
- 不改预测 CSV、不改权重内容（瘦身仅移除冗余 ema/model 副本，保留的 `use=ema` 态不变）。

## 与 3 模型包的关系

3 模型包（`submit_pkg/`，452 MB）是合规子集上限；本单模型包是其进一步降级，预期排序能力略低，
但延迟余量最大。国赛若用本包重跑隐藏测试集，以本包分数为准；若只校验格式/可复现性，仍以上线最佳 CSV 为准。
