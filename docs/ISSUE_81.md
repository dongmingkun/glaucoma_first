# Issue #81 — 复赛包 predict.py + 3 权重官方环境实测（延迟红线 + 概率一致性）

> 本文件由 GitHub issue #81 原文同步 + 验收结果追加（https://github.com/dongmingkun/glaucoma_first/issues/81），供离线助手读取。
> **最终状态（2026-10-21）**：本地 Python 3.12 复测 PASS（输出 / 一致性 / 包体）；**T4 的 p50/p95 仍未实测**，100ms 红线不能据此认证。
> **补遗（2026-10-22，issue #110）**：无 T4 方案落地——新增 `submission_pkg_1model/`（单模型）与 `submission_pkg_2model/`（双模型）两个降级包，默认交付其一；原 3 模型包降为「有 T4 时」候选；新增 `tools/measure_latency.py` 供队长在 5060 上实测安全边际。**T4 栏仍写「未实测」。**

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

---

## 验收结果（2026-10-21，队长本地 Python 3.12 复测）：PASS（T4 待验证）

### 环境（官方版本组合，非 Docker / 非 T4）

| 项 | 实测值 |
|---|---|
| Python | 3.12.10 |
| torch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| CUDA / cuDNN | 12.8 / 91002 |
| GPU | NVIDIA GeForce RTX 5060 Laptop 8GB（**非 T4**；本机无 T4 / Docker） |

### 包内实际文件（终审计量口径）

| 文件 | 大小 | md5(8) | 元信息 |
|---|---|---|---|
| `predict.py` | 14,474 B | `aa5803b3` | 多线程预处理 + 有界预取 |
| `cb384.pt` | 176,314,205 B | `b7aabbdd` | convnext_base / 384 / cache512 / use=model |
| `cnxb_384_all_pseudo.pt` | 176,318,962 B | `96c91286` | convnext_base / 384 / cache512 / use=ema |
| `cs_hi512_all_ra.pt` | 99,831,694 B | `2553135d` | convnext_small / 512 / cache512 / use=ema |
| **合计** | **452.48 MB**（431.5 MiB） | — | ≤ 500MB ✅ |

> ⚠️ 上文工单原文表格里的 352.6MB / 199.7MB 是 **Release 资产**大小；包内 `cnxb` / `cs_hi512` 是进一步瘦身的单份推理权重（176.3MB / 99.8MB）。终审必须按包内实际文件计量，引用 Release 资产大小会误判为超 500MB。

### 输出校验

| 项 | 结果 |
|---|---|
| 行数 / 顺序 | **5855 行**，`Image_name` 与 `test.csv` 完全一致 ✅ |
| 概率 | [0.000034, 0.998450]，无 NaN ✅ |
| label | `Pred_Label == (prob≥0.5)` 全对，阳性率 4.82% ✅ |
| 前 5 行 | 0.040469 / 0.001957 / 0.001529 / 0.001396 / 0.000980，与参考逐位一致 ✅ |
| 输出 md5 | `e34adfe1`；同环境连续两次全量运行**逐位相同** ✅ |
| 跨版本一致性 | 与 Python 3.11 参考（300 张子集）max abs diff **1.21e-4**，0 张 >1e-3 ✅ |

> 与参考 md5 `95a871c7` 不同，属 FP16 跨版本非确定性 / 行尾格式差异；概率逐行校验满足 ≤1e-3，结论不受影响。

### 延迟（全量 5855 张，3 模型 TTA1，bs32，prep-workers 8）

| 轮次 | 总耗时 | mean | 稳态 p50 | 稳态 p95 |
|---|---|---|---|---|
| #1 | 304s | 51.99 ms | 37.54 ms | 77.24 ms |
| #2 | 338s | 57.86 ms | 45.46 ms | 80.66 ms |
| 2 模型对照（复测） | 381s | 65.53 ms | 38.39 ms | 120.66 ms |

- 3 模型两轮 p50 **37.5–45.5 ms**、p95 **77.2–80.7 ms**，本地均 <100ms。
- 测法：端到端按 bs32 逐批计时（预处理等待 + 3 模型前向 + 张量拷贝），每批耗时除以批大小；剔除前 2 个 warmup 批。计时脚本为 `predict.py` 的副本，仅插入 `time.perf_counter`，不改推理逻辑；两次运行的预测 CSV 逐位相同。
- 2 模型那一轮反而更慢（p95 120ms），说明本机 p95 受共享 GPU（其它进程）/ 70W 功耗墙 / 热状态影响，**不能用本机 p95 外推 T4，也不能按模型数线性推断延迟**。
- **T4 的 p50/p95 未实测**（当前无 T4 环境）。因此 100ms 红线只能标记为「本地 5060 通过、T4 待验证」，不能宣称官方 T4 合规。

### T4 复测命令（拿到 T4 后原样执行）

```bash
python predict.py --images <test_images_dir> --csv <test.csv> \
  --weights cb384.pt cnxb_384_all_pseudo.pt cs_hi512_all_ra.pt \
  --blend 0.5 0.25 0.25 --tta 1 --batch-size 32 --prep-workers 8 \
  --out submission.csv
```

### 结论

| # | 验收项 | 结论 |
|---|---|---|
| 1 | 官方版本组合（py3.12 / torch2.10 / tv0.25 / CUDA12.8） | ✅ PASS（本地复现，非官方 Docker） |
| 2 | 5855 行 / 三列 / 范围 / NaN / label | ✅ PASS |
| 3 | p50/p95 ≤100ms | ⚠️ **PARTIAL**：5060 p50 37.5–45.5ms、p95 77.2–80.7ms；**T4 未实测** |
| 4 | 概率一致性 ≤1e-3 | ✅ PASS（前 5 行 0；300 张 max 1.21e-4） |
| 5 | 包体 ≤500MB | ✅ PASS（452.48MB，按包内实际文件） |

**最终：PASS（本地） / T4 待验证。** 本工单以「除 T4 实测外全部通过」关闭；若后续拿到 T4，按上面的命令补测并把 p50/p95 回填到本文件与 `submit_pkg/README.md`；若 T4 p95 >100ms，改用 2 模型配置并重新实测（不要用 5060 的 2 模型数字推断）。

---

## 降级交付包（issue #110，2026-10-22）

### 背景
官方红线 T4 单图 ≤100ms，但无 T4 可认证（租 T4 成本/流程不划算）。决定不租 T4，改用「保守配置 + 安全边际」交付，并明确标注 T4 未实测。

### 交付物
- `submission_pkg_1model/`：单模型（cnxb_384_all_pseudo），predict.py + fp16 权重 + README + measure_latency.py + assemble.py + MANIFEST.md。5060 p95 目标 ≤40ms（2.5× 余量）。
- `submission_pkg_2model/`：双模型（cb384 + cnxb_384_all_pseudo，blend 0.65/0.35）。5060 p95 目标 ≤50ms（2× 余量）。
- `tools/measure_latency.py`：逐批计时，剔除前 2 warmup 批，输出 mean/p50/p95/p99（per-image ms），并给出 5060 上可跑的 `predict.py` 命令。复用 `submit_pkg/predict.py` 的预处理/模型加载，不改推理逻辑。
- 更新 `submit_pkg/README.md`：默认交付单/双模型，3 模型仅作 T4 候选，**T4 一律写「未实测」**。

### 权重来源与瘦身
两个降级包复用 Release `weights-1016` 的现有权重：
- `cb384.pt` ← `convnext_base_384_cache512_1015_auprc0.8644.pt`（已是瘦身单态 fp16，~168MB）。
- `cnxb_384_all_pseudo.pt` ← `cnxb_384_all_pseudo_fp16.pt`（336MB 双态）→ 本地 torch CPU 瘦身单态（保留 `use=ema`，fp16，~176MB），与 3 模型包内同权重一致。
- 瘦身仅移除冗余 ema/model 副本，**不改权重内容**；不占云 GPU，纯 CPU 操作。

### 包体校验（预期，实测见各包 MANIFEST.md）
- 单模型目录：cnxb ≈176MB ≤200MB ✅
- 双模型目录：cb384 ≈168MB + cnxb ≈176MB ≈344MB ≤400MB ✅

### 安全边际验收（队长在 5060 上执行）
- 单模型 p95 ≤40ms 或 双模型 p95 ≤50ms → 视为通过安全边际。
- ⚠️ 5060 p95 受共享 GPU / 70W 功耗墙影响，**不能外推 T4**；T4 100ms 红线须真机认证后才可宣称合规。

### T4 补测（非阻塞）
若后续拿到免费 T4（Kaggle/Colab/队友），按各包 README 的 `measure_latency.py` 命令 10 分钟补测，回填 p50/p95 到本文件与 `submit_pkg/README.md`；若 T4 p95 >100ms，改用单模型配置。
