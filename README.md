# 青光眼二分类挑战赛（眼底彩照）

眼底彩照 → 青光眼阳性概率的二分类任务。**完整的过程、踩过的坑与最新结论都在 [PROJECT_NOTES.md](PROJECT_NOTES.md)，先读它。**

## 当前成绩 / 现状
- 基线 **B3@384 + 年龄融合**（4x 翻转 TTA）：valid **AUROC 0.98504 / AUPRC 0.8159**
- 首版提交（`submission.csv`）排行榜得分 **89.2**（该分数 ≈ valid 上 0.5 阈值的平衡准确率 88.83，而远低于 AUROC 98.5 → 评分口径可能阈值敏感，见 Issue）
- 进行中：B3@448 微调、多模型集成、评分口径判定、复赛镜像实测

## 仓库结构
| 路径 | 说明 |
|---|---|
| `train.py` | 训练（EfficientNet-B3/B4 + 年龄 MLP 融合、EMA、AMP、`--init-from` 热启动、`--train-on-all` 全量重训） |
| `evaluate.py` | 单 checkpoint 验证集评估（支持 `--tta`、`--no-age` 消融） |
| `infer.py` | 测试集推理出提交 CSV（4x 翻转 TTA） |
| `ensemble.py` | 多模型概率平均（`--eval` 验证集增益 / `--submit` 出提交） |
| `calibrate_submission.py` | 提交概率阈值校准（单调 shift，不改排名，只把 0.5 对准 valid 最优点） |
| `preprocess.py` / `verify_cache.py` | 眼底裁剪 → 512 缓存预处理与校验 |
| `audit_data.py` / `md5_dupes.py` | 数据审计、重复图检测 |
| `submit_pkg/` | 复赛自包含提交包（`predict.py` + 权重 + 说明） |
| `glaucoma_first/` | 官方数据 CSV（**图片目录不入库**，从比赛平台下载） |
| `bench*.py` | 吞吐/内存基准脚本 |

## 快速开始
环境：Python 3.11 + torch 2.10.0+cu128 + torchvision 0.25.0（详细安装方法与坑见 PROJECT_NOTES）。

```bash
# 验证集评估（4x TTA）
python evaluate.py --ckpt <best.pt> --tta 4

# 集成评估（看增益）
python ensemble.py --ckpt runA/best.pt runB/best.pt --eval --tta 4

# 出提交
python infer.py --ckpt <best.pt> --split test --tta 4 --out submission.csv
```

## 硬约束（务必遵守）
- **本机内存**：16GB RAM + 页面文件 ≈ 48GB 提交上限；每个 torch 进程 ≥1.25GB。**任何时刻只跑一个 GPU 任务**；训练 `--workers 3`，否则 WinError 1455 崩溃
- **蒸馏/复现**：GPU 功率墙 70W，B3@384 训练 ~41 张/秒（~21 min/epoch）
- **复赛目标环境**：T4 16G，单图端到端 **≤100ms**，权重 **≤500MB**，torch 2.10.0+cu128 / Python 3.12
- **提交配额**：**每天 1 次**（首次报错可补 1 次，约 1 小时出分；以官方页面为准），截止 2026-11-06 —— 任何提交前先在 Issue 登记并在 `SUBMISSIONS_LOG.md` 记录

## 协作方式
- 任务清单与分工都在 **Issues** 页；开始前先认领对应工单，完成后回填数字与结论
- 改动走分支 + PR（`main` 保持可用）；实验结论（无论成败）都写进对应 Issue
- 大文件（权重、数据、缓存）不入库，通过网盘/本机目录分发
- 提交台账与权重命名规范见 `SUBMISSIONS_LOG.md`（#6）；权重命名 `{arch}_{size}_{月日}_ep{epoch}_valid{auroc}.pt`