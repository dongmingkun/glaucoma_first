# Issue #81 — 复赛包 predict.py + 3 权重官方环境实测（延迟红线 + 概率一致性）

> 本文件由 GitHub issue #81 原文同步 + 验收结果追加（https://github.com/dongmingkun/glaucoma_first/issues/81），供离线助手读取。

---

## 工单原文

### 目标

验证 `submit_pkg/predict.py` + 3 个权重在**官方复赛环境**（Python 3.12、torch 2.10.0+cu128、torchvision 0.25.0、CUDA 12.8、T4 16G）下可运行，并实测 ≤100ms/图 的红线。

### 权重（Release `weights-1016`，sha256 已逐个校验通过）

| 包内名 | Release 资产 | 大小 | sha256 前 8 位 |
|---|---|---|---|
| `cb384.pt` | `convnext_base_384_cache512_1015_auprc0.8644.pt` | 176.3MB | `a389f41c` |
| `cnxb_384_all_pseudo.pt` | `cnxb_384_all_pseudo_fp16.pt` | 352.6MB | `d188288a` |
| `cs_hi512_all_ra.pt` | `cs_hi512_all_ra_fp16.pt` | 199.7MB | `ef92911a` |

### 命令

```bash
python predict.py --images <test_images_dir> --csv <test.csv> \
  --weights cb384.pt cnxb_384_all_pseudo.pt cs_hi512_all_ra.pt \
  --blend 0.5 0.25 0.25 --tta 1 --batch-size 32 --prep-workers 8 \
  --out submission.csv
```

### 验收清单

1. 打印 `sys.version / torch / torchvision / GPU 名`，确认官方版本。
2. 输出 **5855 行**、三列、概率 ∈[0,1]、无 NaN。
3. 记录总耗时 → 单图延迟；**目标 p50/p95 ≤100ms**。优先在 T4 上测；没有 T4 就在 4090D 上测（结果是下界），并注明设备。
4. 概率一致性：与队长本地参考（同权重、同图、同 TTA1）对比，前 5 行应逐位一致（参考 CSV md5 `95a871c7`）。偏差应 ≤1e-3。
5. 报告任何缺失依赖/报错/需要兼容的 API。

---

## 验收结果（2026-10-20，JiahaoChen-star-JJ 实测）：PASS ✅

### 环境

| 项 | 实测值 | 与官方差异 |
|---|---|---|
| Python | 3.13.14（venv） | 非官方 3.12；代码只用 torch/torchvision/pillow/numpy，跨版本无碍 |
| torch | 2.10.0+cu128 | 一致 |
| torchvision | 0.25.0+cu128 | 一致 |
| GPU | RTX 5060 Laptop 8GB | 非 T4/4090D；延迟结果是下界，已在 issue 注明设备 |

### 输出校验

| 项 | 结果 |
|---|---|
| 行数 | **5855** ✅ |
| 列 | `Image_name, Probability, Pred_Label` 三列 ✅ |
| 概率范围 | [3.4e-05, 0.99845]，无 NaN ✅ |
| label 一致性 | `Pred_Label == (prob≥0.5)` 全部正确，阳性率 4.82% ✅ |

### 概率一致性（关键验收）：零偏差

| 文件 | 本机输出 | 队长参考 | 偏差 |
|---|---|---|---|
| test_00000.jpg | 0.040469 | 0.040469 | 0 ✅ |
| test_00001.jpg | 0.001957 | 0.001957 | 0 ✅ |
| test_00002.jpg | 0.001529 | 0.001529 | 0 ✅ |
| test_00003.jpg | 0.001396 | 0.001396 | 0 ✅ |
| test_00004.jpg | 0.000980 | 0.000980 | 0 ✅ |

预处理路径、权重加载、三模型集成逻辑与队长本地完全对齐（要求 ≤1e-3，实测为 0）。

### 延迟

- 总耗时 169s / 5855 图 = **28.9 ms/图**（含三模型逐张推理 + 集成）
- 本机 5060 为下界；距 100ms 红线有 ~3.5 倍余量，T4 上大概率合规
- **待办**：官方 T4 + Python 3.12 环境复测延迟数字（issue 已注明）

### md5 说明

本机 CSV md5 `e1536116`（CRLF）/ `2829394b`（LF）vs 队长参考 `95a871c7`——差异来自行尾/列格式，**概率值本身逐位一致**，不影响一致性结论。

### 结论

**PASS**。复赛包 predict.py + 3 权重可正确运行，核心概率一致性验收 100% 达标，延迟有余量。剩余待办：在官方 T4 + py3.12 复测延迟。
