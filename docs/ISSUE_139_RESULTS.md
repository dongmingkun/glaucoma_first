[ISSUE_118_RESULTS.md](https://github.com/user-attachments/files/33974426/ISSUE_118_RESULTS.md)
# Issue #118/#139 结果记录 — Swin-B@384 16ep × 3-fold clean OOF + test（#127 异构 fold 成员）

> 状态：**已执行完成**（本机 RTX 5060 Laptop 8GB，非云端 24GB；详见「执行与事故记录」）。
> 流水线等价 `tools/pipeline_118.sh`（用 `local_run139.sh` 本地适配：bs4×accum12、按折续跑、先校验后落盘），
> 含 PR #131（OOF 固定 `last.pt + --use ema`）与 PR #136（空 Age 同口径清洗、失败即停）的全部防线。

## 配置

| 项 | 值 |
|---|---|
| 命令 | `PYTHON=<venv-python> bash local_run139.sh`（等价 `CACHE_SRC=cache512 BS=8 ACCUM=6 bash tools/pipeline_118.sh`） |
| 机器 | **本机 RTX 5060 Laptop 8GB**（bs 8→4 / accum 6→12，等效 batch 48；未占队长 5060，但**非 24GB 云端**，见注意事项） |
| 耗时 | fold0 6.2h（10-25 15:41→21:54）、fold1 9.4h（10-25 23:05→10-26 08:32）、fold2 6.8h（10-26 16:33→23:23），合计约 22.4h（含两次人工中断与重启） |
| 缓存 | `cache512 51396/5849/5855` → `cache_all`（**硬链接** 57245/57245/5855，非符号链接，Windows 无符号链接权限） |
| 口径 | 每折 ImageNet 初始化、seed 42；OOF/test 一律 `last.pt` + 固定 `--use ema`（禁止 best/auto） |

## 每折 held-out valid TTA4（19082/19082/19081 行，各 616 正例）

| fold | AUROC | AUPRC | balACC@0.5 |
|---|---|---|---|
| 0 | 0.98054 | 0.76687 | 0.8038 |
| 1 | 0.98394 | 0.80955 | 0.8200 |
| 2 | 0.98503 | 0.81939 | 0.8134 |

## OOF 57245 整体（#127 clean stack 的异构 fold 成员）

- 覆盖：**57245/57245**（3 折无重叠，脚本先校验后落盘）
- **OOF AUROC：0.98304**
- **OOF AUPRC：0.79645**
- OOF balACC@0.5：0.8124

> 参考 #126（Small fold base OOF 0.98685）：本 Swin 折 OOF 略低（-0.0038），符合"折训练集仅 38163（全量 51396 的 74%）"的预期；
> 本成员的定位是**架构多样性**而非绝对分。

## 折间多样性（test 概率两两 Spearman，门槛 <0.98）

| 对 | Spearman |
|---|---|
| fold0 × fold1 | **0.8136** |
| fold0 × fold2 | **0.7495** |
| fold1 × fold2 | **0.8020** |

**全部远低于 0.98 → 多样性验收通过**（同架构仅换数据划分即达此相关性，主要来自折间样本完全不相交 + Swin 大 lr 训练的随机性）。

## official valid 5849 子集：vs ens9（blend w 扫描 + 配对 bootstrap 2000 次）

ens9 基线取队长 team_valid 包（`ens9_valid_tta4_raw.csv`，**AUROC 0.99260372** 复现验证一致，5849 行顺序与官方 valid 逐条一致）。
fold-ensemble 在 official-valid 段的概率 = OOF 概率的子集（对 fold 模型全 held-out；ens9 成员在该段也干净，双侧口径成立）。

- fold-ensemble(swin3) official-valid：**AUROC 0.98386 / AUPRC 0.80625**
- ens9 official-valid：**AUROC 0.99260**

| w | blend AUROC | Δ vs ens9 | P(Δ>0) | CI95 |
|---|---|---|---|---|
| 0.05 | 0.99249 | **-0.00012** | 0.037 | [-0.00027, +0.00001] |
| 0.10 | 0.99238 | -0.00022 | 0.028 | [-0.00051, +0.00000] |
| 0.15 | 0.99227 | -0.00033 | 0.018 | [-0.00070, -0.00002] |
| 0.20 | 0.99211 | -0.00050 | 0.006 | [-0.00100, -0.00008] |
| 0.25 | 0.99193 | -0.00068 | 0.004 | [-0.00131, -0.00016] |
| 0.30 | 0.99175 | -0.00085 | 0.003 | [-0.00159, -0.00023] |
| 0.40 | 0.99139 | -0.00121 | 0.001 | [-0.00222, -0.00039] |
| 0.50 | 0.99096 | -0.00164 | 0.000 | [-0.00292, -0.00059] |

**结论：无增益（best w = 0）**。任何正权重都让 valid AUROC 下降；w=0.05 时 Δ=-0.00012、P(Δ>0)=0.037，
远未达门槛（Δ≥+0.0005 / P≥0.90）。

对照 #126（Small fold base 单独 vs ens9 亦为 best w=0）：**同架构换数据划分的 fold 成员，对 ens9 均无 blend 增益**——
说明瓶颈不在"折内数据量"，而在 Swin@384 与 ens9 成员的**精度代差**（0.9839 vs 0.9926，差 0.0087）。
#127 的希望仍需看 #134（RA Base fold）这类**基座精度接近**的异构成员。

## 验收清单（对照 #139）

- [x] OOF 覆盖 57245、3 折无重叠（`OOF cover=57245/57245 OK`，先校验后落盘）
- [x] OOF/test 固定 `last.pt + --use ema`（每折推理日志 `use=ema`，checkpoint 为 `last.pt`）
- [x] test 5855 行 / 官方顺序 / 无 NaN（3 折平均，`test rows 5855 test_mean 0.0495`）
- [x] 3 折两两 Spearman <0.98（0.8136 / 0.7495 / 0.8020）
- [ ] `runs/swin_fold{0,1,2}/last.pt` 挂 Release/群文件（668MB × 3，待上传）
- [ ] 是否进 #127 stack：由 #127 nested CV gate 决定（提分 Δ≥+0.0005/P≥0.90；99 档 Δ≥+0.0015/P≥0.95），不在本单判定

## 执行与事故记录

1. **环境适配**：本机 8GB 显存，`--batch-size 4 --accum 12`（等效 48）；`cache_all` 用 Windows 硬链接（`os.link`）构建
   57245/57245/5855，不占额外磁盘。显存峰值约 7.0GB（训练）/ 10.3GB commit（含 WDDM 共享），训练未 OOM。
2. **fold0/fold1 由队友（Zcode 会话）完成**（10-25 15:41 / 10-25 23:05 起），本会话接手 fold2。
3. **fold2 三次启动**：第一次跑到 ep8（10-25，人工暂停）、第二次跑到 ep3（10-26 14:45，人工暂停）、第三次
   16:33 起跑至 23:23 完成。**每次中断后必须删除 `runs/swin_fold2/{last,best}.pt`**，否则 `local_run139.sh`
   的跳过条件（`last.pt` 存在）会用未跑满的权重直接推理，污染 OOF 口径——本单严格执行，最终 OOF 三折均为 16ep。
4. **踩坑修复**：`local_run139.sh` 原 `CACHE="/d/glaucoma/cache_all"` 为 POSIX 路径，Windows python 会拼出
   `/d/glaucoma/cache_all\train\...` 报 FileNotFoundError；改为 `CACHE="${CACHE:-D:/glaucoma/cache_all}"` 后正常
   （该修复已在本地脚本中，建议同步回 `tools/pipeline_118.sh` 的 Windows 分支说明）。
5. **其余**：无 HF 下载（swin_base 权重已在缓存）、无校验失败、无 NaN。

## 独立复算

- 工具：本会话 `tools/analyze_139.py`（sklearn AUROC/AP + scipy Spearman + 自写配对 bootstrap 2000 次、seed=0），
  与队友 `tools/oof_report.py`（团队口径）可交叉对照；
- 复算范围：6 个 CSV 行数/顺序/NaN、OOF 组装一致性（57245 覆盖无重复）、每折与 OOF 指标、折间 Spearman、
  official-valid 子集 blend（独立 bootstrap 种子）。
- 复算命令：
  ```bash
  cd repo_main3 && python tools/analyze_139.py --ens9 <ens9_valid_tta4_raw.csv>
  ```
