# Issue #3 — train+valid 全量重训（`--train-on-all`）

> 目标：把 valid 的 5849 张（+11.4% 数据，其中 191 个阳性）并进训练集，换约 +0.002 AUROC。
> 代价：**这个 checkpoint 再没有干净的验证集**，所有"跑 evaluate.py 看分数"的直觉在这里都失效。

---

## 1. 什么时候该做

**只在最后交付前做一次。** 它是一个不可逆的单向门：

- 做完之后你无法再判断这个模型好不好（valid 已被训练过）
- 它不能用来做消融、不能用来选 epoch、不能用来调阈值
- 收益只有 +11% 数据对应的那一点，风险却是"交了个没验证过的东西"

前置条件全部满足再动手：

| 条件 | 检查方式 |
|---|---|
| 已有 train-only 的 best.pt 作热启动 | `runs/<exp>/best.pt` 存在且带 val 指标 |
| 已确定 train-only 的最佳 epoch 数 N | 看 `runs/<exp>/log.csv` 的 val 曲线拐点 |
| `cache512` 三个 split 完整 | `python verify_cache.py` → TOTAL BAD: 0 |

从零训（不热启动）做 train-on-all 是**明确错误的做法**：没有 val 曲线兜底，过拟合无从察觉。

---

## 2. epoch 数怎么定（最容易踩的坑）

`--train-on-all` 后每个 epoch 的样本数从 51396 变成 57245，**optimizer step 数随之 ×1.114**。
沿用 train-only 的 epoch 数 = 多训 11.4% 的 step，正好是本项目的翻车点（B3@512 就是多训了就过拟合，AUPRC 一路从 0.81 掉到 0.74）。

**按总 step 对齐，不要按 epoch 对齐：**

```
epochs_on_all = round(epochs_train_only / 1.114)
```

| train-only 最佳 epoch | 全量重训应取 |
|---|---|
| 5 | 4 |
| 8 | 7 |
| 10 | 9 |

`train.py` 启动时会打印这条换算提示，照抄即可。宁少一个 epoch，不要多一个。

---

## 3. 命令

```bash
VENV=D:\glaucoma_work\venv\Scripts\python.exe

$VENV train.py \
  --arch convnext_tiny --img-size 384 \
  --epochs 7 \
  --batch-size 24 --accum 2 --loader workers --workers 3 \
  --init-from "D:/glaucoma_work/runs/<train_only_exp>/best.pt" \
  --train-on-all --tag all_final
```

架构选择：现在的主力是 **ConvNeXt-Tiny**（笔记 2026-10-15：单模 4xTTA AUROC 0.98945 / AUPRC 0.85584，已超过 B3 三方集成），不要再拿 B3 做全量重训。

产物在 `D:\glaucoma_work\runs\<exp>\`：

- `best.pt` **和** `last.pt` 内容相同（最后一个 epoch），都写出来了 —— 因为下游
  `make_submission.py` / `ensemble.py` / 交接文档一律按 `best.pt` 取，以前只写 `last.pt` 会导致产物接不上
- checkpoint 里带 `train_on_all=True` 和 `use="ema"`，下游不再靠"两个 -1 比较"猜该用哪套权重
- 权重固定用 EMA（train-only 的每一轮 EMA 都不差于 model，无 val 时这是唯一站得住的选择）
- `log.csv` 正常记录 train_loss / lr / 耗时（以前全量模式完全不写，等于盲训）

---

## 4. 验收：用 `check_train_on_all.py`，不要用 evaluate.py

**全量模型的 valid AUROC/AUPRC 是假的**（valid 在训练集里，指标必然虚高，通常能刷到 0.99+）。
`infer.py` 和 `make_submission.py` 检测到 `train_on_all` 标记时会直接打警告。

唯一干净的检查是**和 train-only 参考模型在 test 上的一致性**：

```bash
$VENV check_train_on_all.py \
  --ckpt "D:/glaucoma_work/runs/all_final/best.pt" \
  --ref  "D:/glaucoma_work/runs/convnext_384/best.pt" \
  --tta 4
```

判定（两条都要满足）：

| 指标 | 阈值 | 含义 |
|---|---|---|
| test 概率 Spearman 相关 | ≥ 0.90 | 排序没崩（Spearman 对单调变换免疫，校准过的概率也能比） |
| mean prob 相对偏差 | ≤ 25% | 概率尺度没整体漂移 |
| top200 重叠率 | 参考值 | AUPRC 靠头部，这个数掉了说明头部质量变差 |

不达标就别交：通常是 epoch 多了（fold 进 valid 后更容易过拟合），退一个 epoch 重跑。

---

## 5. ⚠️ 校准陷阱

`make_submission.py --calibrate` 是用 **valid** 拟合 logit shift 的。对全量模型来说 valid 被污染，
概率系统性偏高，拟合出的 shift 会把阈值推到错误位置 —— 校准在这里不是帮忙，是帮倒忙。

正确做法：**校准系数始终来自 train-only 模型**。

```bash
# 1) 用 train-only 模型拿到干净的 shift（看它打印的 calibration shift）
$VENV make_submission.py --ckpt runs/train_only/best.pt --tta 4 --calibrate --out /tmp/ref.csv

# 2) 把同一个 shift 施加到全量模型的 test 概率上
$VENV calibrate_submission.py --in submission_all_raw.csv --shift <上一步的 shift> --out submission_final.csv
```

（第 2 步的脚本参数以 `calibrate_submission.py --help` 为准。）

---

## 6. 提交前

- 每天只有 **1 次**提交配额，先在 Issue #3 登记，再交
- 提交后回填 LB 分数 + `check_train_on_all.py` 的 SUMMARY 行
- 台账记进 `SUBMISSIONS_LOG.md`

回填模板（一行）：

```
exp=all_final arch=convnext_tiny img=384 epochs=7 init_from=<exp> | spearman=0.9xx mean_rel=0.0xx top200=0.8xx | LB=xx.xx
```

**不要回填 valid 的 AUROC/AUPRC** —— 那个数字会被后来的人当成真实指标引用，污染整条决策链。

---

## 7. 本次改动清单（Issue #3）

| 文件 | 改动 |
|---|---|
| `paths.py`（新增） | 统一的路径解析：环境变量 / 仓库布局 / 旧硬编码三级回退。`CSV_ROOT` 和 `CACHE_ROOT` 原本写死队长机器的 `C:\Users\Kevin\...`，**在任何其他机器上 `--train-on-all` 都会直接 FileNotFoundError**，这是工单的第一道阻塞。抽成无 torch 依赖的独立模块，`verify_cache.py` 之类不需要为找个路径就 import torch。 |
| `train.py` | ① 用 `paths.py` 解析路径，新增 `--csv-root` / `--cache-root`；② 全量模式不再构建 val DataLoader（省下 worker 的 ~1.25GB/进程提交内存，这条是本机铁律）；③ 同时写 `best.pt` 和 `last.pt`；④ checkpoint 补 `use` / `train_on_all` / `val_metrics_meaningful` 字段；⑤ 全量模式照写 `log.csv`；⑥ 启动时打印 step 换算提示和"val 指标作废"警告 |
| `infer.py` `ensemble.py` `make_submission.py` | 尊重 checkpoint 的 `use` 字段（原来靠 `-1 >= -1` 的巧合选中 EMA，改字段就会翻车）；遇到 `train_on_all` 且评估 train/valid 时打污染警告；`make_submission.py` 的 report 增加 `train_on_all` 标记 |
| `verify_cache.py` | 改用 `paths.py` |
| `check_train_on_all.py`（新增） | 全量重训唯一可行的验收手段 |

未做（有意为之，避免越界）：没有改 `MixedDS`/`FundusDS` 的增强实现，没有动损失函数与采样策略。
