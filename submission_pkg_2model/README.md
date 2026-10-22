# 降级提交包（双模型）— submission_pkg_2model

> 对应工单 **#110**：无 T4 方案。本包是「保守配置 + 安全边际」交付的**备选双模型**选项。
> **T4 未实测**（当前无 T4 环境）；下方延迟数字均为 RTX 5060 实测/估算，仅作下界参考。

## 内容（双模型，包体 ≤400 MB）

| 文件 | 大小 | 说明 |
|---|---|---|
| `predict.py` | 14 KB | 自包含推理脚本（与 `submit_pkg/predict.py` 主版本逐行一致） |
| `cb384.pt` | ~168 MB | ConvNeXt-Base@384，train-only，cache512，fp16 单态 |
| `cnxb_384_all_pseudo.pt` | ~176 MB | ConvNeXt-Base@384，train+all + 测试域伪标签，cache512，fp16 单态 |
| `measure_latency.py` | 6 KB | 逐批计时工具（mean/p50/p95，剔除前 2 warmup 批） |
| `assemble.py` | 3 KB | 从 Release `weights-1016` 拉权重 + 瘦身 cnxb |
| `MANIFEST.md` | — | 权重 md5 / 元信息 / 包体校验记录 |

权重元信息（由 `MANIFEST.md` 实测）：两者均为 `arch=convnext_base` / `img_size=384` /
`cache_size=512`；`cb384.pt` 用 `use=model`，`cnxb_384_all_pseudo.pt` 用 `use=ema`。

## 推荐命令（TTA1，双模型 blend 0.65/0.35）

```bash
python predict.py \
  --images /path/to/test_images \
  --csv /path/to/test.csv \
  --weights cb384.pt cnxb_384_all_pseudo.pt \
  --blend 0.65 0.35 \
  --tta 1 --batch-size 32 --prep-workers 8 \
  --out submission.csv
```

## 延迟与安全边际（5060，下界；**T4 未实测**）

- 目标：5060 上 **p95 ≤ 50 ms**（相对 100 ms 红线的 2× 余量）。
- blend 0.65/0.35：`cb384` 占 0.65、`cnxb_384_all_pseudo` 占 0.35（train-only 单模更稳，给更高权重）。
- ⚠️ 5060 的 p95 受共享 GPU / 70W 功耗墙影响，**不能外推到 T4**；T4 的 100 ms 红线须真机认证。

```bash
python measure_latency.py \
  --images /path/to/test_images --csv /path/to/test.csv \
  --weights cb384.pt cnxb_384_all_pseudo.pt --blend 0.65 0.35 \
  --tta 1 --batch-size 32 --prep-workers 8
```

## 组装 / 校验

```bash
python assemble.py --mode 2model     # 拉 cb384 + 瘦身 cnxb（需 torch；否则给预瘦身地址）
```

- 包体：双模型目录 ≤400 MB ✅（cb384 ≈168 MB + cnxb ≈176 MB ≈344 MB）。
- `predict.py` 与 `submit_pkg/predict.py` 主版本一致，仅复制、未改推理逻辑。
- 不改预测 CSV、不改权重内容。

## 与 3 模型包的关系

双模型包排序能力介于单模型与 3 模型之间，延迟余量（p95 ≤50 ms）优于 3 模型（p95 ~80 ms）。
3 模型包仅作为「有 T4 时」的候选，不作为默认交付（见 `submit_pkg/README.md`）。
