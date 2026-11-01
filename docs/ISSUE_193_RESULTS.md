# #193 ConvNeXt-Large@384 + RA + train+all seed7（Kaggle T4×2）

> 状态：**已交付**。两场 Kaggle 会话跑完 16 epoch，产出 `cnxl_384_e16_ra_all_s7_test_tta4.csv`（5,855 行），10 项格式复核 PASS。
> **未提交平台**；按工单要求，三人 CSV 汇总后统一评估。

## 0. 结论

| 项 | 值 |
|---|---|
| 模型 | `convnext_large` @384，RandAugment(2,9)，train-on-all（含 5,849 例 valid），seed 7 |
| 训练 | 16 epoch / 19,072 优化步；会话① ep1–8（8.00 h，SIGTERM 落盘），会话② ep9–16（6.95 h，exit=0） |
| 算力 | Kaggle T4 单卡（T4×2 只用一张），合计 14.95 h，周额度 30 h |
| 交付 | `cnxl_384_e16_ra_all_s7_test_tta4.csv`，5,855 行，158,120 B |
| 指纹 | SHA256 `7169871037cda31b2a87459d5237d944d913b253e0f2986bde81243dfc804af8`；MD5 `122f9eb31beb6797896cb93b1b68b8af` |
| 复核 | 10/10 PASS（`docs/audit_193/audit_seed7.json`） |
| 与 98.87 的关系 | Spearman 0.644，top200 命中 178/200；按 w=0.05 做 rank 融合，与本单 98.87 最终排序的 Spearman 0.9993 |

## 1. 两次会话时间线

### 1.1 为什么改成 headless 脚本内核

第一次在交互式 notebook 里跑到 `ep1 it2350` 崩掉，报 `RuntimeError: basic_ios::clear: iostream error`，紧跟 `unexpected pos 2301136704`，最后 nbconvert `OSError: [Errno 28] No space left on device`。
**不是 OOM，是 `/kaggle/working` 被写满**：之前几次 smoke 各自留下 2×1.57 GB 的 `last/best.pt`，再加 3.15 GB `resume.pt` 和同样大小的 `.tmp`，20 GB 配额当场见底。

改法：换成 `kernel_type: script` 的批量内核（干净容器、正常退出即自动发布产物），`--save-every-steps 0` 只在 epoch 末落盘。实测峰值约 9.4 GB，余量充足。

### 1.2 会话①（内核 v1）

| epoch | train_loss | 耗时 | 结束 lr |
|---|---|---|---|
| 1 | 0.4039 | 3475 s | 5.00e-05 |
| 2 | 0.2264 | 3432 s | 4.92e-05 |
| 3 | 0.2013 | 3427 s | 4.75e-05 |
| 4 | 0.1842 | 3423 s | 4.48e-05 |
| 5 | 0.1697 | 3407 s | 4.12e-05 |
| 6 | 0.1573 | 3410 s | 3.70e-05 |
| 7 | 0.1434 | 3407 s | 3.23e-05 |
| 8 | 0.1383 | 3401 s | 2.73e-05 |

8 小时预算到点，runner 发 SIGTERM，`train.py` 在 epoch 9 第 2796 个 micro-batch（`global_step=10002`）落盘 `resume.pt`（2.94 GB）后退出，`train exit=143`。

### 1.3 会话②（内核 v2）

`resume.pt` 先下载到本地，再以私有数据集 `linglingyusc/g193-resume-ep8-s7` 挂回 `/kaggle/input`（脚本内核的 `/kaggle/working` 不跨会话保留，这是与 notebook 最大的区别）。

| epoch | train_loss | 耗时 | 结束 lr |
|---|---|---|---|
| 9 | 0.1119 | 3109 s | 2.22e-05 |
| 10 | 0.0943 | 3122 s | 1.72e-05 |
| 11 | 0.0858 | 3117 s | 1.26e-05 |
| 12 | 0.0747 | 3111 s | 8.49e-06 |
| 13 | 0.0684 | 3106 s | 5.11e-06 |
| 14 | 0.0610 | 3113 s | 2.58e-06 |
| 15 | 0.0585 | 3115 s | 1.03e-06 |
| 16 | 0.0538 | 3120 s | 5.00e-07 |

`[runner] train exit=0 elapsed=6.95h`，随后自动推理：`wrote cnxl_384_e16_ra_all_s7_test_tta4.csv (5855 rows)`。

单 epoch 51.8–57.9 min，**T4 上 16 epoch 约 14.9 h**，正好落在"两周额度、两场会话"的区间；原工单按 40 min/epoch 估的 10.7 h 偏乐观。

## 2. 断点续跑的证据

```
[resume] /kaggle/input/g193-resume-ep8-s7/resume.pt: 已完成 8/16 epoch，epoch 9 跳过 2796 个 batch，global_step=10002
```

- LR 跨会话连续：ep8 结束 2.73e-05 → ep9 开始 2.22e-05，没有回弹（只有从 epoch 整数重开才会让 `cos(pi·p)` 越界反弹）。
- 剩余步数严格等于 `total_steps - global_step`：19,072 − 10,002 = 9,070 步，对应 7.6 个 epoch。
- 代价：被跳过的 2796 个 micro-batch 仍要解码（不解码无法定位），实测约 6 分钟，相对 7 小时可忽略。

## 3. 交付件校验

`tools/audit_193_seed7_csv.py` 对仓库内交付件实跑，`verdict: PASS`：

| # | 检查 | 结果 |
|---|---|---|
| 1 | 行数 = 5,855 | PASS |
| 2 | 表头 `Image_name/Probability/Pred_Label` | PASS |
| 3 | 行序与 `glaucoma_first/test.csv` 逐行一致 | PASS |
| 4 | 文件名唯一 | PASS |
| 5 | 全部 finite | PASS |
| 6 | 概率 ∈ [0,1] | PASS（min 0.000000 / max 0.998341） |
| 7 | `Pred_Label` 与概率符号一致 | PASS（不一致 0 行） |
| 8 | CRLF | PASS |
| 9 | 无 BOM | PASS |
| 10 | SHA256 记录 | PASS |

`mean_prob = 0.043301`，`pos_rate = 0.042186`（与测试集正例率同量级）。
**这是 train-on-all 的 BCE 输出，概率尺度没有和融合成员校准过**——所以只能走 rank 口径，不要拿它和既有成员做 logit 平均。

同时复核了与 98.87 复现件的一致性基线：`docs/audit_98_87/FINAL_REPRO_98_87.csv` 的 MD5 仍为 `4bf37b7c062f7d45a9c5a8e287fff1eb`，98.87 本身未被本单改动。

## 4. 与现有成员的关系（0.64 的 Spearman 不是问题）

| 参照 | Spearman | top50 | top100 | top200 |
|---|---|---|---|---|
| 98.87 融合件 | 0.6440 | 42/50 | 85/100 | 178/200 |
| seed123 · cnxl@384（同配方换种子） | 0.6776 | 42/50 | 85/100 | 177/200 |
| seed123 · cnxb@512 | 0.6471 | — | — | 180/200 |

作为对照，98.87 与 seed123-cnxl384 之间的 Spearman 是 **0.8508**。三者互相关都略低于这一个数，说明 s7 确实带来了一点新信息，而不是复制品。

全局 Spearman 偏低的原因是**分布形状**，不是排序错乱：

- 只看 top-200 内部，s7 与 98.87 的排序相关 **0.8457**，与 seed123-cnxl384 **0.8204**；
- 98.87 与 seed123 的 top-100 交集共 88 例，其中 **87 例落在 s7 的 top-200 内**；
- s7 的头部更"尖"：top-100 的最小概率 0.9917，基线 0.9353，中低位的近零样本压走了全局秩相关。

## 5. 融合模拟（动 98.87 的最小方式）

以 98.87 成品 `B` 为底，`new = (1-w)·rank(B) + w·rank(s7)`：

| w | 与 B 的 Spearman | B 的 top-100 仍在新 top-100 | B 的 top-100 仍在新 top-200 |
|---|---|---|---|
| 0.05 | 0.99931 | 99/100 | 100/100 |
| 0.10 | 0.99716 | 99/100 | 100/100 |

也就是说 w=0.05/0.10 只在尾部做微调，符合"失败不得低于 98.87"的约束。
`Pred_Label` 在 0.5 阈值下会翻转 63 / 142 行，但赛题按 `Probability` 算 AUROC×100，`Pred_Label` 不参与评分；代码侧仍保证两者自洽（见第 3 节第 7 项）。

**收益只能由平台反馈证明**，离线没有验证集（train-on-all 已吃掉 valid），不要把这组数字当成涨分预测。

## 6. 交付物

| 路径 | 内容 |
|---|---|
| `cnxl_384_e16_ra_all_s7_test_tta4.csv` | 测试集概率，5,855 行 |
| `docs/audit_193/audit_seed7.json` | 10 项复核 + SHA256 + 与三个参照的 Spearman/top200 |
| `docs/ISSUE_193_RESULTS.md` | 本文件 |
| `train.py` | `--resume-from` / `--save-every-steps` / 原子断点 / 信号处理（前序提交） |
| `infer.py` | `--cache-root` / `--csv-root` 覆盖（前序提交） |
| `tools/check_193_env.py` | 开跑前预检（前序提交） |
| `tools/audit_193_seed7_csv.py` | 交付 CSV 复核（前序提交） |
| `tools/kaggle_193_runner.py` + `tools/kaggle_193_kernel.json` | 本单实际使用的 headless runner 与内核元数据 |

## 7. 复现跑法

```bash
# 1) 把 runner 和 metadata 放进一个空目录，按需改 id / dataset_sources
# 2) 标准 kaggle 工作流
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python -m kaggle kernels push -p <dir>
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python -m kaggle kernels status linglingyusc/g193-cnxl384-s7
# 3) 结束时拉产物
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python -m kaggle kernels output linglingyusc/g193-cnxl384-s7 -p out \
  --file-pattern 'cnxl_384_e16_ra_all_s7_test_tta4\.csv$'
```

runner 做的事情：找 `cache_all` → 拉 `main` 分支 → 装 timm/sklearn/pandas/pillow → 递归找 `resume.pt`（`/kaggle/input/<slug>/` 和 `/kaggle/input/datasets/<owner>/<slug>/` 两种挂载都认）→ 跑 16 epoch（8 h 预算到点 SIGTERM）→ 有 `best.pt` 就 `infer.py --tta 4` → 打印 `/kaggle/working` 全部文件体积。

注意：批量内核 **RUNNING 期间 `kernels logs` 是空的**，只能看 `kernels status`；日志在结束后才可取。交互式 notebook 的 `/kaggle/working` 在报错后会被丢弃，别在那里做长跑。

## 8. 合规与红线

- 数据：`linglingyusc/glaucoma` 保持**私有**；眼底图未设为 public；未使用他人私有 Dataset；训练只读 `cache_all`，未用 test 标签拟合任何东西。
- 提交：**未提交平台**，三人 CSV 汇总后由队里统一评估。
- 凭据：Kaggle legacy key 只放在本机 `~/.kaggle/kaggle.json`，未写入仓库、日志或提交信息；用完建议吊销。

## 9. 仍未解决

1. **封包 500 MB 红线**：现有 3 模型包 475.82 MB，加一个 Large（fp16 约 350–390 MB）到 826–868 MB，超限约 70%；用 Large 顶掉 cb384 也要约 650 MB。要"替换谁 / 全 INT8 / 只留离线结论"需队长先拍板——本单不替队长决定。
2. **本 seed 的绝对分**只有平台能给；离线只能给"与既有成员的排序关系"和"融合扰动幅度"。
3. 周额度已用 14.95 h（会话①+②），如果还要补第三场 seed，需重新算账。
