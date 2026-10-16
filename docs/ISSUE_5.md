# Issue #5 — 复赛镜像实测（py3.12 + T4）+ predict.py 多模型支持（完成记录）

> 多模型 / 多架构 / 烘焙校准已由 **PR #18** 交付，预处理去重 + 张量翻转 + draft 对齐已由
> **PR #21** 交付（均已合并）。本 PR 只保留一处独有增量：**`load_model()` 尊重 checkpoint 的
> `use` 字段**，并更新实测清单。实测侧（T4 / Python 3.12）仍在官方镜像待执行。

## 1. 本 PR 独有改动（一处）

`submit_pkg/predict.py` 的 `load_model()`：

```python
use = ckpt.get("use")
if use not in ("ema", "model"):
    use = "ema" if ckpt.get("auroc_ema", -1) >= ckpt.get("auroc", -1) else "model"
```

原因：`train-on-all` 权重（PR #16 引入）带 `use: ema` + `val_metrics_meaningful: False`，
其 `auroc_ema`/`auroc` 都是 -1。旧逻辑 `-1 >= -1` 靠巧合选中 ema，一旦字段变更就翻车。
统一后，`predict.py` 与 `make_submission.py / infer.py / ensemble.py` 的选权口径一致。

## 2. 延迟红线（#20 实测 / #21 结论，与 submit_pkg/README.md 一致）

| 配置 | GPU 延迟 | 预处理 | 单图端到端 | 判定 |
|---|---|---|---|---|
| 3 模型 + TTA4 | ~115ms | ~40ms | ~155ms | ❌ 超标 |
| 2 模型 + TTA4 | ~85ms | ~30ms | ~115ms | ⚠️ 临界 |
| **2 模型 + TTA1（推荐）** | ~35ms | ~30ms | **~75ms** | ✅ 合规 |

推荐复赛配置（`--tta 1`，必要时 `--calib-shift` 烘焙校准）：

```bash
python predict.py \
  --images /path/to/test_images \
  --csv /path/to/test.csv \
  --weights cs_hi512_v2.pt cs_hi512_v2.pt cnxb.pt --blend 1 1 1 \
  --tta 1 --calib-shift <logit常数> --out submission.csv
```

降级顺序（若实测 p95 仍超 100ms）：`--tta 4→1` → 减少集成成员 → 降低输入分辨率。

## 2.5 多尺度 TTA 延迟预算（#43，2026-10-16）

多尺度 TTA（#30 已证实 Small@512 三尺度 AUPRC +0.0054）会把每模型前向成本 × 尺度数，需在 ≤100ms 红线内做定量取舍：

| 候选配置 | 端到端估算 | 判定 | 说明 |
|---|---|---|---|
| 单模型 × 3 尺度 × TTA1 | **~82ms** | ✅ 可行 | 吃满多尺度收益（+0.0054 AUPRC） |
| 2 模型 × 1 尺度 × TTA1（现有推荐） | ~75ms | ✅ 合规 | 稳健档，收益略低 |
| 2 模型 × 3 尺度 × TTA1 | ~135ms | ❌ 超红线 | 成员 × 尺度双重开销 |

**结论**：多尺度收益只能在「少成员 + 低 TTA」下使用。推荐两档：
- **提分优先**：单模型 hi512_v2 × 3 尺度 × TTA1（~82ms）
- **稳健优先**：2 模型 × 1 尺度 × TTA1（~75ms）

`make_submission.py` 已加 `--img-sizes`（多尺度语义与 `evaluate.py` 一致）；预估 LB 用 #29 复合口径公式在提交流程内直接输出。

## 3. 实测清单（官方镜像内执行，待权重 + logit 常数就绪）

```bash
# 0) 环境自检
python -c "import sys,torch,torchvision; print(sys.version.split()[0], torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))"
# 期望: 3.12.x  2.10.0+cu128  0.25.0+cu128  NVIDIA T4

# 1) 推荐配置（2 模型 + TTA1 + 烘焙校准）
python predict.py --images /path/test_images --csv /path/test.csv \
  --weights cs_hi512_v2.pt cs_hi512_v2.pt cnxb.pt --blend 1 1 1 \
  --tta 1 --calib-shift <logit常数> --out submission.csv
```

### 验收清单

| # | 项 | 通过标准 |
|---|---|---|
| 1 | 版本 | Python 3.12、torch 2.10.0+cu128、torchvision 0.25.0 |
| 2 | 设备 | `torch.cuda.get_device_name(0)` == NVIDIA T4 |
| 3 | 权重体积 | 多模型合计 ≤500MB（fp16 瘦身见 docs/WEIGHTS.md） |
| 4 | 单图延迟 | 推荐配置 p50/p95 ≤100ms/img |
| 5 | 预处理一致性 | 去重 + 张量翻转后概率与逐图计算偏差 ≤1e-3 |
| 6 | 输出格式 | 三列、行数 = test 行数、概率 ∈[0,1]、无 NaN |
| 7 | 选权一致性 | train-on-all 权重与 infer.py 选同一套（ema），排序一致 |

## 4. Issue 回填文案（直接粘贴到 #5）

```
#5 返工完成（以 main 已合并的 #18/#21 为准，只带回 use 字段增量）

- load_model() 尊重 ckpt 的 use 字段: train-on-all 权重(use:ema, auroc=-1)不再靠 -1>=-1 巧合选权,
  与 make_submission.py / infer.py / ensemble.py 口径统一
- 多模型/多架构/烘焙校准(#18)、预处理去重+张量翻转+draft对齐(#21) 已在 main, 本 PR 不重复
- 延迟红线对齐 README: 3模型TTA4 ~155ms 超标 / 2模型TTA4 ~115ms 临界 / 2模型TTA1 ~75ms 推荐
- 实测侧待执行(官方镜像 T4/py3.12): 推荐配置 2模型+TTA1+烘焙校准, 验收清单见 docs/ISSUE_5.md
```

## 5. 本地预验收（#33，2026-10-16，无 T4 可做部分）

> 复赛包 `submit_pkg/predict.py` 在无 T4 条件下可做的全部预验收，把风险提前挤出。

### 已通过项（本机静态 + 逻辑验证）

| # | 项 | 结果 | 依据 |
|---|---|---|---|
| 1 | py3.12 语法兼容 | ✅ | `py_compile` 通过；无 3.12 移除/弃用 API（distutils / imp / 旧 typing 别名等均未出现） |
| 2 | 依赖自包含 | ✅ | 仅 `torch / torchvision / PIL / numpy`，官方镜像全部自带 |
| 3 | 裁剪档位三级回退（#28） | ✅ | `infer_cache_size` 逻辑单测 6 例全过：L1 `cache_size` → L2 `args.cache_root` 含 768 → L3 文件名含 `cache768` → 默认 512 |
| 4 | 体积预算 | ✅ | 最优集成 fp16 瘦身 **424MB ≤ 500MB**（余量 ~76MB，见 docs/WEIGHTS.md） |
| 5 | 命名规范 | ✅ | `--out` 默认 `submission.csv`，符合平台命名要求 |
| 6 | 输出格式 | ✅ | 三列 `Image_name,Probability,Pred_Label`；`Pred_Label = (p>=0.5)`；概率裁剪在 [0,1] |

### 必须 T4 / 权重就绪才能做的剩余项

| # | 项 | 阻塞原因 |
|---|---|---|
| 1 | 瘦身权重实际加载（20 张小样本格式自检） | 需 Release fp16 权重（weights-1016）+ 解压后 test 图片 |
| 2 | 延迟实测 p50/p95（≥100 张） | 需 T4 环境；4090D 实测无法按 0.7–1.0x 外推 T4 |
| 3 | 官方镜像多模型 + 烘焙校准端到端 | 需 T4 / py3.12 / torch 2.10.0+cu128 官方镜像 |
| 4 | 概率一致性 ±1e-3（去重+张量翻转 vs 逐图） | 需权重 + 数据 + GPU |

### 结论

本地可做的预验收全部通过，**未发现需提 PR 的兼容性问题**（predict.py 已封板：#18 多模型 / #20 延迟优化 / #28 三级回退均在 main）。剩余 4 项为「待 T4 / 待权重」的运行时验证，已列入上表。
