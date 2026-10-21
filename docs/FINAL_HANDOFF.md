# 最终交付物审计 + 交接（FINAL_HANDOFF）

> 关联工单：#97（P0，无 GPU）。本文件由 @Yulingling-sc 在 2026-10-21 起草。
> 说明：**§1 提交包审计 / §2 候选 CSV 复现**依赖队长本地 `D:\glaucoma_work\submissions\` 与 `D:\glaucoma_work\tmp\gh_issues\`，当前环境（本地 D 盘 + 云端）均不可达，**暂未完成**，标记为 TODO；本版完成 **§3 台账一致性 / §4 交接文档 / §5 PR**。待队长提供 D: 文件后补 §1/§2。

---

## 一、§3 台账一致性审计（基于云端 `SUBMISSIONS_LOG.md`）

> 数据源：云端 `/root/autodl-tmp/glaucoma_first/SUBMISSIONS_LOG.md`（最新可用版本）。

### 审计结论

| #97 §3 列出的 ID=分数 | 云端台账是否记录 | 一致性 |
|---|---|---|
| 12700 = 89.20 | ✅ 已记录（2026-10-14，@dongmingkun，`6303c195`） | 一致 |
| 12762 = 94.07 | ✅ 已记录（2026-10-15，ens4_cal，`2f065a39`） | 一致 |
| 13050 = 94.81 | ❌ **云端台账缺失该条目** | 无法核验 |
| 13420 = 98.15 | ❌ **云端台账缺失该条目** | 无法核验 |
| 14098 = 98.82 | ❌ **云端台账缺失该条目** | 无法核验 |
| 14386 = 98.80 | ❌ **云端台账缺失该条目** | 无法核验 |
| 15022 = 98.73 | ❌ **云端台账缺失该条目** | 无法核验 |
| 15720 = 98.81 | ❌ **云端台账缺失该条目** | 无法核验 |

**发现（FAIL）**：云端 `SUBMISSIONS_LOG.md` 仅记录至 2026-10-16（最高分 94.07，ens7 计划提交行）。#97 §3 所指的 98.xx 高分提交（94.81–98.82，对应 ID 13050–15720）**全部不在云端台账中**。原因推测：队长本地有更完整的 `SUBMISSIONS_LOG.md`（含后期提交回填），云端这份是早期快照。

**待办**：需队长提供本地最新 `SUBMISSIONS_LOG.md`（或平台回执截图），方可完成 §3 全量核对（日期 / 文件 / md5 / 分数是否一致）。

---

## 二、§4 最终交接文档

### 2.1 当前最佳

- **线上最佳：98.82（ens10）**（来源：#97 工单描述；**注意**：该分数未出现在云端台账，需 §3 回填后确认）。
- 候选次优：ens20_swin = 98.81、ens21_swin2 = 10-22 备选。

### 2.2 候选清单（文件 / md5 / 预期 / 状态）

| 候选 | 提交文件 | md5(前8) | 预期/实际分 | 状态 |
|---|---|---|---|---|
| **ens10** | `submission_20251018_ens9_cal.csv` 衍生（un-shift 常数 1.4587） | `ce91bbf4` | **98.82（当前最佳）** | 已提交 |
| **ens20_swin** | `submission_20251021_ens20_swin_cal/` | `d9c3b212` | 98.81 | 已提交 |
| **ens21_swin2** | `submission_20251022_ens21_swin2_cal/` | `fb4a750a` | 10-22 备选 | 备选（待提交） |

> md5 / 分数为 #97 工单给定；§2 复现校验（用成员 test 概率 + ens9 基线重算并比对 md5）**待 D: 文件到位后执行**。
> 成员 test 概率（cs_hi512_all、cs_hi512_all_ra、Swin 阶段1 / stage2 v3 等）已上传 GitHub Release `weights-1016`（`*_test_probs_tta4.csv`），可用于 §2 复现。

### 2.3 提交步骤（平台规则，源自 `SUBMISSIONS_LOG.md` #6 规范）

1. **只上传 `submission.csv`**：平台要求的上传名为 `submission.csv`（最终包内追溯名 `<候选名>.csv` + `manifest.md` + `package/`）。
2. **每天 1 次成功配额**：提交前先登记「计划提交」行（文件名 + md5 + 来源 + 预期分），提交后 10 分钟内回填 LB 分数与结论，避免重复提交浪费配额。
3. **评分耗时 5–15h**：提交后耐心等待；最终取历史最高分。
4. **不手工改 CSV**：要改就改 checkpoint/权重，重跑 `make_submission.py`。

### 2.4 复赛包状态（源自 `submit_pkg/README.md` + PR #89/#81）

- **包内容**：`predict.py`（自包含推理，仅依赖 torch/torchvision/pillow/numpy）+ `model_weights.pt`（含 ema/model 双套 + 元信息）。
- **环境**：GPU NVIDIA T4 16GB；CUDA 12.8 + cuDNN 9；Python 3.12；PyTorch 2.10.0 + torchvision 0.25.0。
- **官方环境实测**：PR #89（#81）确认 `predict.py` 在官方镜像内跑通（PASS），本地 RTX 5060 下界延迟 28.9ms/图，距 100ms 红线 3.5 倍余量。
- **延迟红线（≤100ms 单图端到端）**：
  - 3 模型 + TTA4 ≈ 155ms ❌ 超标；
  - 2 模型 + TTA4 ≈ 115ms ⚠️ 临界；
  - **2 模型 + TTA1 ≈ 75ms ✅ 合规（推荐）**。
  - #97 提及「3 模型 TTA1」配置，其 T4 实测延迟仍属 **待测**（submit_pkg README 标注「待 T4 环境」）。
- **权重合计**：fp16 瘦身成员合计 < 500MB（限制见 `docs/WEIGHTS.md`）。

### 2.5 已知风险与红线

- **严禁使用外部医学预训练 / 外部标签**：仅用比赛官方数据 + ImageNet-1k/22k 公开初始化。
- **口径红线**：线上分数口径为「阈值类（平衡准确率@0.5，校准有效）」与「排序类（混合 0.5×(AUROC+AUPRC)，校准无效）」两候选并存（见 `SUBMISSIONS_LOG.md` #1 判定）；提交前务必确认候选落在哪一档、校准是否生效。
- **延迟红线**：复赛单图端到端 ≤ 100ms；优先 2 模型 + TTA1 合规配置。
- **配额红线**：每天仅 1 次成功提交，提交前必登记、提交后必回填。
- **§1/§2 阻塞**：提交包 bundle 物理审计与候选 CSV 复现 md5 校验，待队长提供 `D:\glaucoma_work\submissions\` 与 `D:\glaucoma_work\tmp\gh_issues\` 后完成。

---

## 三、§1 / §2 待办（TODO，阻塞于队长本地文件）

- [ ] **§1 Bundle 审计**：逐项检查 `submission_20251021_ens20_swin_cal/`、`submission_20251022_ens21_swin2_cal/` —— 根目录 `submission.csv`、内部 `<候选名>.csv` / `manifest.md` / `package/` 是否齐全；CSV 5855 行 / 三列 / 无 NaN / 概率∈[0,1] / `Image_name` 顺序与 `glaucoma_first/test.csv` 一致；`package/` 权重合计 ≤500MB。
- [ ] **§2 候选 CSV 可复现性**：用 `D:\glaucoma_work\tmp\gh_issues\` 成员 test 概率 + `submission_20251018_ens9_cal.csv`（un-shift 1.4587）复现 ens10/ens20_swin/ens21_swin2，核对 md5（ce91bbf4 / d9c3b212 / fb4a750a）。
- [ ] **§3 全量核对**：用队长最新 `SUBMISSIONS_LOG.md` 补全 98.xx 条目与 #97 §3 列表的逐行比对。

> 提供上述 D: 文件（或云可达路径）后，本文件将补齐 §1/§2 并移除 TODO。
