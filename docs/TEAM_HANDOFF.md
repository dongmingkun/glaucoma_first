# 团队上手文档（TEAM HANDOFF）

> 给两位队员的工作指引。配合 GitHub Issues #1–#8 使用。
> 维护者：队长（@dongmingkun）｜ 最后更新：2026-10-14 深夜

## 0. 当前战况（必读）

| 项目 | 现状 |
|---|---|
| 线上最好成绩 | **89.20 分**，第 34 名（2026-10-14 21:48 提交，纯 CSV，1 小时内出分） |
| 榜首 | 96.05 分；前 14 名 92.2–96.05 |
| 本地验证集（5849 张，阳性 3.27%） | 最佳单模 AUROC **0.98504** / AUPRC **0.81592**；384+448 集成 AUROC 0.98507 / AUPRC 0.82839 |
| 提交规则 | **每天只有 1 次**（首次报错才能补 1 次），1 小时出分；截止 2026-11-06 |

### ⚠️ 有一条尚未定论的关键线索（影响一切优化方向）
线上 89.2 与本地 AUROC 98.5 / AUPRC 81.6 都对不上。已排除：跨集合泄漏、图像分布漂移、患者级聚合、准确率类指标。剩下两个候选：

1. **平衡准确率 @0.5 阈值**（本地 88.83，最接近 89.2）→ 若成立，提交文件的"概率尺度/阈值位置"决定生死
2. **0.5×(AUROC+AUPRC) 混合**（本地 90.05）→ 若成立，靠集成/更强模型逐步提升

**已备好"判别实验"**：`submission_20251015_ens_cal.csv` 是把集成概率做了单调重映射（让最优工作点落在 0.5）。对排序类指标**分数完全不变**，对阈值类指标**预计 +5 分**。提交后看分数即可判定口径：
- 跳到 ~94 → 是候选 1（阈值类）
- 停在 ~90.7 → 是候选 2（排序类）

## 1. 环境搭建（每台新机器一次）

```bash
# 需要：NVIDIA GPU（≥8GB 显存）、Python 3.11、~60GB D 盘空间
# 1) 拉代码
git clone https://github.com/dongmingkun/glaucoma_first.git && cd glaucoma_first

# 2) 队员自己从比赛平台下载数据集，放到 glaucoma_first/glaucoma_first/{train,valid,test}/
#    以及同名 csv：train.csv / valid.csv / test.csv

# 3) 建 venv（注意：本机 ensurepip 有坑，用 get-pip 引导）
py -3.11 -m venv --without-pip D:\glaucoma_work\venv
curl -sL -o D:\glaucoma_work\get-pip.py https://bootstrap.pypa.io/get-pip.py
D:\glaucoma_work\venv\Scripts\python.exe D:\glaucoma_work\get-pip.py

# 4) 装 torch（与赛题服务器一致；PyPI 上的 torch 是 CPU 版，必须下 cu128 轮子）
#    官方 CDN 约 4MB/s，支持断点续传；下载后本地安装
D:\glaucoma_work\venv\Scripts\python.exe -m pip install scikit-learn pillow
#    torch 2.10.0+cu128 / torchvision 0.25.0+cu128 轮子见 PROJECT_NOTES.md「本机环境」

# 5) 生成 512 缓存（约 15 分钟，8 进程）
D:\glaucoma_work\venv\Scripts\python.exe preprocess.py      # 产物在 D:\glaucoma_work\cache512
D:\glaucoma_work\venv\Scripts\python.exe verify_cache.py    # 校验：应无 missing / bad
```

## 2. 铁律（踩过的坑，别再踩）

1. **同一台机器任何时候只跑一个 GPU 任务**。系统提交内存上限 ≈48GB，一个训练进程吃 10-11GB，每个 DataLoader worker 吃 1.25GB，并发必崩（症状：`WinError 1455 页面文件太小` / `Couldn't open shared file mapping` / 训练卡死几十分钟）。
2. 训练固定用 `--loader workers --workers 3`（不要用 threads，GIL 卡死在 22 张/秒；不要开 channels_last，本机会慢 5 倍）。
3. 所有产物（venv、缓存、checkpoint、提交）放 **D 盘**，C 盘留给系统。
4. checkpoint 命名规范：`D:\glaucoma_work\runs\<arch>_<size>_<时间>_<tag>\{best,last}.pt`；**交付物 = best.pt + 一行验证集指标**。
5. 崩溃后用 `--init-from <best.pt>` 热启动，别从头训。

## 3. 一键生成提交（唯一入口）

```bash
VENV=D:\glaucoma_work\venv\Scripts\python.exe

# 单模型：算验证集指标 -> 测试集预测 -> 校准 -> 写 CSV -> 校验
$VENV make_submission.py --ckpt runs/A/best.pt --tta 4 --calibrate --out submission.csv

# 多模型集成（权重可选，默认等权）
$VENV make_submission.py --ckpt runs/A/best.pt runs/B/best.pt --weights 1 0.8 --tta 4 --calibrate --out submission.csv
```
它会打印：每个模型的 AUROC/AUPRC/balACC、集成的四种口径预期分（AUROC / AUPRC / 混合 / 平衡准确率）、校准前后对比、以及 CSV 格式校验结果；同时生成 `submission_report.json` 供台账登记。

**任何人不要手工改 CSV**。要改就改 checkpoints 或权重，重跑这条命令。

## 4. 交付物约定（每个训练工单）

| 交付 | 路径 | 说明 |
|---|---|---|
| 权重 | `runs/<exp>/best.pt` | 训练脚本自动产出；注明 arch / img_size / 验证指标 |
| 验证集概率 | `runs/<exp>/probs_valid_full_tta4.npy` | `evaluate.py --ckpt <best.pt> --tta 4` 自动产出（顺序与 valid.csv 一致） |
| 指标报告 | Issue 回复 | 一行：`exp / arch / img_size / AUROC / AUPRC / balACC@0.5 / epoch 用时 / 训练机配置` |
| 权重文件怎么共享 | GitHub Release 附件 或 群文件 | **不要直接 commit 到仓库**（93MB/个，会撑爆 repo） |

## 5. 提交台账（#6 工单维护）

每次提交后在 Issue #6 追加一行：

```
| 日期 | 提交人 | 文件名 | 内容（模型/集成/校准） | 本地预期分（四口径） | 线上分数 | 结论 |
```

线上分数是唯一的最终裁判。任何"预期分"都只是本地估计，务必如实登记。

## 6. 工单索引（与 PROJECT_NOTES.md 一致）

| Issue | 优先级 | 负责人 | 状态（2026-10-14 深夜） |
|---|---|---|---|
| #1 判定评分口径 + 校准 A/B | P0 | @JiahaoChen-star-JJ | **产物已就绪**（`submission_20251015_ens_cal.csv`），待提交 |
| #2 B3@448 微调 + 与 384 集成 | P0 | 原派 @Yulingling-sc | **已完成**（448: AUPRC 0.8267；集成: AUROC 0.98507/AUPRC 0.82839）→ 该队友改做 #3/#4 |
| #3 train+valid 全量重训 | P1 | @Yulingling-sc | 待开始（命令见第 7 节） |
| #4 B4@384 / B4@448 对比 | P1 | @Yulingling-sc | 待开始 |
| #5 复赛镜像实测 + predict.py 多模型 | P0 | @JiahaoChen-star-JJ | `submit_pkg/predict.py` 已单模型验证；多模型支持待做 |
| #6 提交台账 | P2 | @JiahaoChen-star-JJ | 建立 |
| #7 from-scratch 合规备选 | P2 | @JiahaoChen-star-JJ | 待开始 |
| #8 建模探索池 | P2 | @Yulingling-sc | 待开始（512 分辨率微调已在队长机器上跑，见 PROJECT_NOTES 记录） |

## 7. 常用命令速查

```bash
VENV=D:\glaucoma_work\venv\Scripts\python.exe

# 训练（384，从零）：约 21 分钟/epoch，8 epoch 建议
$VENV train.py --arch efficientnet_b3 --img-size 384 --epochs 8 --batch-size 24 --accum 2 \
  --loader workers --workers 3 --tag b3_384_new

# 训练（448 微调，从已有 384 权重热启动）：约 15 分钟/epoch
$VENV train.py --arch efficientnet_b3 --img-size 448 --epochs 4 --batch-size 16 --accum 3 \
  --lr 3e-5 --head-lr 1.5e-4 --init-from runs/A/best.pt --tag b3_448b

# train+valid 全量重训（#3，无验证集，按 epoch 曲线定 epoch 数）
$VENV train.py --arch efficientnet_b3 --img-size 384 --epochs 6 --batch-size 24 --accum 2 \
  --train-on-all --init-from runs/A/best.pt --tag b3_all

# 单模型评估（自动存 probs_valid_full_tta4.npy）
$VENV evaluate.py --ckpt runs/A/best.pt --tta 4

# 集成对比（只看指标，不出提交；改 compare_ensembles.py 里的路径即可）
$VENV compare_ensembles.py

# 出提交（唯一入口）
$VENV make_submission.py --ckpt runs/A/best.pt runs/B/best.pt --tta 4 --calibrate --out submission.csv
```

### 两条血泪教训（2026-10-15 补充）
1. **checkpoint 选择标准**：`--select-metric` 默认已是 `blend`（= 0.5*(AUROC+AUPRC)）。原因：512 分辨率那次，只按 AUROC 选出的 best.pt 恰好是 AUPRC 最差的 epoch（AUPRC 0.77 vs 最优 0.8112，差 0.04）——被选中的模型在混合/阈值类口径下很吃亏，而选择依据的差异（AUROC 0.9821 vs 0.9816）根本在噪声内。
2. **分辨率与过拟合**：448 相对 384 提升 AUPRC；但 512 从 448 热启动后 4 个 epoch 里 AUPRC 一路下滑（过拟合）。**高分辨率微调建议只跑 1-2 epoch**，并盯住验证集 AUPRC 曲线。

## 8. 规矩

- 提交机会只有 1 次/天，**用之前必须在 Issue #1/#6 登记**：谁提交、哪个文件、本地预期分。
- GitHub 上所有实验（含失败的）都要在对应 Issue 留结论，失败也是有价值的记录。
- 遇到内存/环境问题先看 PROJECT_NOTES.md 的「已知坑」，别重复踩。