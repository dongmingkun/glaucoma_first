# 青光眼二分类挑战赛 — 项目笔记

## ❌ 已验证无效的方向（别再重复试）
1. **患者双眼概率混合**（连续同年龄配对，w=0.1~0.5）：全部使指标变差（w=0.1 时 AUPRC 0.8731→0.8676）。原因：官方指标逐图计算，单图预测比双眼平均更准。见 `test_pair_blending.py`
2. **512 缓存内的分辨率提升**（384→448→512 微调）：ConvNeXt 家族 384≈448；B3 只在 448 有小幅 AUPRC 收益。→ 需要 768 缓存才能验证"缓存本身是否瓶颈"（进行中）
3. **准确性类/患者级聚合口径**：与线上分数不符（已排除）
4. **在线程里做数据加载**（ThreadLoader）：GIL 限制，只有 22-25 张/秒
5. **channels_last**：本机慢 5 倍

## 🔑 线上评分口径悬案（2026-10-14，优先级最高，影响所有优化方向）
> 提交台账（**权威数据源**）：`SUBMISSIONS_LOG.md`；探针脚本：`analysis_metric_probe.py`
线上首版 89.20（1x TTA 概率，无校准）。与本地数字对不上：本地 valid AUROC 0.98504 / AUPRC 0.81592。
**已排除**：跨集合患者泄漏（按"连续同年龄=同患者"分组的对照实验，无泄漏组 AUROC 反而更高 0.9853 vs 0.9806）；图像分布漂移（train/valid/test 的原生尺寸、亮度、视野占比分布几乎一致）；患者级聚合 AUROC（99.2-99.6）；准确率类指标（全预测阴性就能拿 96.7 分，与榜单 31-35 名在 88.8-90 矛盾）；F1/MCC（量级差太远）。
**剩下两个候选**（本地值，与榜单 89.2 的差距）：
1. **平衡准确率 @0.5 阈值 = 88.83**（Δ0.37，且在 191 阳性的抽样误差内）← 最强候选
2. 0.5×(AUROC+AUPRC) = 90.05（Δ0.85）
**为什么关键**：模型训练用了 pos_weight=5，概率被系统性推高，0.5 阈值不在最优工作点（最优在 raw≈0.07，那里平衡准确率 94.3）。若榜单是阈值类指标，则**单纯校准就能 +5.5 分**。
**判别实验**：把概率做严格单调重映射（logit 平移），让最优工作点落到 0.5 —— 排序类指标（AUROC/AUPRC/任何排序加权）分数**严格不变**，阈值类指标大幅提升。零风险。
- 产物：`submission_20251015_ens3_cal.csv`（**384+448+512 三方集成** + 校准，2026-10-15 凌晨生成）
  - 本地预期：平衡准确率@0.5 = 0.9438（→ 线上 ~94.4）；0.5×(AUROC+AUPRC) = 0.90894（→ 线上 ~90.9）；AUROC 0.98517；AUPRC 0.83270
- 早期两方版本 `submission_20251015_ens_cal.csv`（384+448）为备份：最高分口径下两者取优，建议提交三方版
- **最强版** `submission_20251015_ens4_cal.csv`（ConvNeXt-Tiny + B3×3 四模型集成 + 校准，见「架构级发现」）：
  - 本地预期：平衡准确率@0.5 = 0.9554（→ 线上 ~95.5）；0.5×(AUROC+AUPRC) = 0.9268（→ 线上 ~92.7）；AUROC 0.98979；AUPRC 0.86374
- **实测（10-15 09:01 提交）：线上 94.07（第 3 名）** → 平衡准确率预测 95.54（差 −1.47）、混合分预测 92.68（差 +1.39）→ **两个候选均未排除**（残差都在测试集抽样噪声 ±1.4 内），需要第三个数据点
- 口径探针脚本：`analysis_metric_probe.py`（#1 任务，在 valid 上逐一口径对比，自动标注最接近 89.2 的候选）
- 校准实现：`make_submission.py --calibrate`（唯一提交入口，内含四口径预期分 + 格式校验）

## 📊 榜单口径分析（两个数据点，2026-10-15 更新）
| 提交 | 内容 | 本地 AUROC | 本地 AUPRC | 混合分 | 校准后平衡准确率 | **线上实测** |
|---|---|---|---|---|---|---|
| S1 (10-14 21:48) | 单模 B3@384 TTA4，未校准 | 0.98504 | 0.81592 | 90.05 | 88.83 | **89.20**（第34名）|
| S2 (10-15 09:01) | ens4 四模型集成 + 校准 | 0.98979 | 0.86374 | 92.68 | 95.54 | **94.07**（第3名）|

**排除**：AUROC（预测 98.5/99.0，残差 −9.3/−4.9）；AUPRC（预测 81.6/86.4，残差 +7.6/+7.7；"测试集阳性率更高"假设经重加权检验不自洽：两次解出 7.7% vs 10.7%）；准确率类（全阴性白卷 96.7 分逻辑矛盾）；患者级聚合（99.2+）。
**存活的两个候选**（残差都在 ±1.5 抽样噪声内）：
1. `0.5×(AUROC+AUPRC)`（残差 −0.85 / +1.39）
2. `平衡准确率@0.5`（校准后，残差 −0.37 / +1.47）
**关键**：两者对下一版（ens6：AUROC 0.99020 / AUPRC 0.86940 / 校准后 balACC 0.95916）的预测几乎一致（≈94.4），策略无需二选一 —— 继续提升模型质量 + 保持校准（排序口径下无害、阈值口径下有益）。

**正例率检查（#1 任务2）**：校准后 Pred_Label 正例率明显高于 valid 阳性率 3.27%，属**预期而非 bug** —— 校准把 Youden 最优点移到 0.5，sens 0.80→0.96、spec 0.98→0.945，预测阳性自然增多（理论正例率 ≈ P·sens + N·(1−spec) ≈ 8.3%，与实测吻合）：

| 提交 | Pred_Label 正例率 | 概率均值 | 概率中位数 |
|---|---|---|---|
| `submission.csv`（原始，未校准）| 4.92% | 0.0568 | 0.0009 |
| `submission_cal.csv`（B3@384 校准）| 10.35% | 0.1233 | 0.0117 |
| `submission_20251015_ens4_cal.csv` | 8.71% | 0.1080 | 0.0119 |
| `submission_20251016_ens6_cal.csv` | 9.29% | 0.1231 | 0.0201 |

## ✅ 校准策略结论（#1 任务4）
1. **以后每次提交统一走校准**（`make_submission.py --calibrate`）：校准是单调 logit 平移，排序类口径（AUROC/AUPRC/混合）下分数**严格不变**，阈值类口径（balACC@0.5）下 **+5 分**。零风险、稳赚不赔，作为默认策略。
2. **换模型/集成必须重算最优点**：各集成的最优点 `raw_t*` 不同（B3@384 单模 0.067 → ens4 0.0837 → ens6 0.0985），换模型后必须用 `calibrate_submission.py` 在 valid 上重新求最优点，**不得沿用旧的 shift 常数**。
3. **口径未锁定前不阻塞**：两个候选对当前最强版（ens6）的线上预测几乎一致，继续提升模型质量即可，无需二选一；第三个数据点（ens6_cal 提交后）锁定口径后回填本表。

## 🚀 架构级发现：ConvNeXt 系列全面超过 EfficientNet-B3（2026-10-15）
同样的数据/增强/损失/训练配方，ConvNeXt-Tiny@384（8 epoch 从 ImageNet 权重从头训）达到：
- 无 TTA：EMA AUROC **0.9873** / AUPRC **0.8392**（B3 最好 0.9849/0.8047）
- 4x TTA：**AUROC 0.98945 / AUPRC 0.85584**（单模就超过 B3 三方集成 0.98517/0.83270）

ConvNeXt-Small@384（8 epoch）：TTA4 **AUROC 0.98994 / AUPRC 0.86371**
ConvNeXt-Base@384（8 epoch）：TTA4 **AUROC 0.99117 / AUPRC 0.86439**（当前最强单模；注意无 TTA 的 blend 上 Base≈Small，TTA4 后才看出优势）

**规模阶梯（AUPRC, TTA4）**：B3 0.8159 → Tiny 0.8558 → Small 0.8637 → Base 0.8644（Base 仅微增，容量接近饱和）
**集成配置（valid, TTA4）**：
| 配置 | AUROC | AUPRC | blend | maxbalACC |
|---|---|---|---|---|
| ens6（Small+0.4Tiny+0.2B3三方） | 0.99002 | 0.86999 | 93.00 | 95.71 |
| **Base+Small+B3三方（1:1:0.5）** | 0.99098 | **0.87310** | **93.20** | 95.88 |
| Base*1.2+Small+Tiny+0.4B3 | **0.99107** | 0.87044 | 93.08 | **96.23** |

**分辨率结论（ConvNeXt 系列已到顶）**：
- ConvNeXt-Tiny：@384 (0.85584) > @448 (0.84973)
- ConvNeXt-Small：@384 (blend 0.9243) ≈ @448 (blend 0.9244)
→ 对 ConvNeXt，384 已经足够；**盲目堆分辨率无效**（B3 不同：384→448 有 +0.011 AUPRC 收益）
→ 结论：512 缓存的可用信息在 384 输入下已基本榨干；继续提升靠 **架构放大**（Tiny→Small→Base）与**更多数据**，而非分辨率

**模型规模阶梯（AUPRC, TTA4）**：B3 0.8159 → Tiny 0.8558 → Small 0.8637 → (Base 训练中)

四模型集成（ConvNeXt-Tiny + B3三方，权重 1:0.5）：
- AUROC 0.98979 / AUPRC 0.86374 / 混合 92.68 / max平衡准确率 95.54 → **线上实测 94.07（第3名）**
- 提交文件：`submission_20251015_ens4_cal.csv`

六模型集成（ConvNeXt-Small + ConvNeXt-Tiny + B3三方，权重 1:0.4:0.2）：
- AUROC 0.99020 / AUPRC 0.86940 / 混合 92.98 / 校准后平衡准确率 95.92
- 提交文件：`submission_20251016_ens6_cal.csv`（预期线上 ~94.4）

**对后续工作的影响**：
- ConvNeXt 家族是主力（#4 的 B4 实验改为 ConvNeXt 系列实验）
- 待做：ConvNeXt-Base@384（已启动）、train-on-all 全量重训、第二个 ConvNeXt-Small（种子多样性）
- 训练成本：Tiny@384 ~21 min/epoch；Small@384 ~25 min/epoch（bs12+accum4，显存 4.3GB）；Base@384 ~40 min/epoch（bs8+accum6，显存 4.7GB）

## 赛题要点
- 任务：眼底彩照 → 青光眼阳性概率（二分类），提交 `submission.csv`（Image_name, Probability, Pred_Label）
- 指标：主 AUROC，次 AUPRC；每天限 2 次提交；截止 2026-11-06
- 数据：train 51396（阳性 3.22%）、valid 5849（阳性 3.27%）、test 5855；附 Age 字段
- 服务器约束（复赛验证）：T4 16G，单图端到端 ≤100ms，模型压缩前 ≤500MB，torch 2.10.0 + torchvision 0.25.0 + CUDA 12.8，Python 3.12
- 规则：仅可用主办方数据；允许数据增强；不允许外部标注数据

## 本机环境
- GPU：RTX 5060 Laptop（8GB，sm_120，需 cu128 轮子）；**功率墙 70W（上限 115W，改不了），实测 B3@384 训练 ~41-64 张/秒**
- CPU 32 逻辑核 / 内存 16GB / 所有产物放 D 盘（`D:\glaucoma_work`）
- venv：`D:\glaucoma_work\venv`（Python 3.11.9，pip 26.2.1 由 get-pip 引导）
- torch：**2.10.0+cu128 + torchvision 0.25.0+cu128**（与赛题服务器完全一致）
- 安装注意：
  - `download.pytorch.org` 的 R2 CDN 间歇 DNS 失败；PyPI 上的 torch 是 CPU 版。**用 curl 从官方 CDN 下载 wheel（支持 -C - 续传）再本地 pip install**，或阿里云 pytorch-wheels 镜像（但只有 0.3MB/s，官方 4MB/s）

## ⚠️ 内存约束（本项目最大的坑）
系统提交内存上限 = 16GB RAM + 31.6GB 页面文件（D 盘，系统托管）≈ 48GB。
- Windows 基线（含 WSL）~14-20GB
- 每个 torch 进程（含 DataLoader worker）**至少 1.25GB 提交内存**；主训练进程前 5 步后稳定在 **~10.4GB**
- 症状：`RuntimeError: Couldn't open shared file mapping ... error code: <1455>` / `OSError: [WinError 1455] 页面文件太小` / 新进程起不来（0xC0000142）；伴随 GPU 掉频、训练卡死几十分钟
- **规则：任何时候只跑一个 GPU 任务**；训练用 `--workers 3`（训练）+ 2（验证，非持久化），总提交 ~31GB，裕度 17GB
- 试过但**不可行**：线程预取 ThreadLoader（PIL 解码释放 GIL 但 torchvision 增广是 Python 级，被 GIL 卡住，只有 22-25 张/秒）
- 崩溃后恢复：`--init-from runs/xxx/best.pt` 热启动（自动选 ema/model 中验证 AUROC 更高的一套）

## 训练速度实测（RTX 5060 Laptop @70W）
- B3@384 bs32 fp16：991 ms/step ≈ 32 张/秒（纯 GPU 上限）
- 完整训练循环 bs24+accum2 + 3 workers：**41 张/秒**（epoch ≈ 21 分钟）
- channels_last 在本机病态地慢 5 倍（5378ms/step），**禁用**
- 推理 B3@384 bs32：5.7 ms/img；T4 上预计也是这个量级，满足 100ms 限制

## 训练结果记录
| 运行 | 配置 | 结果 |
|---|---|---|
| b3_384_base | B3@384, 计划 8ep, 4 workers | ep1 EMA AUROC **0.9672**; ep2 EMA **0.9714**; 之后因内存崩溃中止 |
| b3_384_cont | 从 ep2 best 热启动, 5ep, 3 workers | ep5 **AUROC 0.9849 / AUPRC 0.8047**（EMA 0.9844/0.8037）|
| b3_448 | 从 384 best 热启动, 448px, 4ep | **AUROC 0.9820 / AUPRC 0.7996**（TTA4 后 AUPRC **0.8267**，比 384 高 0.011，但 AUROC 略低）|
| b3_512 | 从 448 best 热启动, 512px, 4ep | ⚠️ **过拟合**：ep1 的 AUPRC 0.8112 最好，之后一路降（0.78/0.77/0.74）而 AUROC 稳定 0.981-0.982。TTA4: AUROC 0.98370 / AUPRC 0.81415。教训：**best.pt 只按 AUROC 选会挑中 AUPRC 最差的 epoch** → 已改 `--select-metric` 默认 `blend`（0.5*(auroc+auprc)）|
| **集成 384+448+512** | 等权概率平均 + 4x TTA | **AUROC 0.98517 / AUPRC 0.83270 / 混合 90.89 / max平衡准确率 94.38**（比两方集成各项都高）|

**分辨率效应（重要）**：448 相对 384，AUPRC +0.011 而 AUROC -0.003 → 更高分辨率显著改善"排序头部质量"，正好补我们的短板（AUPRC）。512 单模反而过拟合（4 epoch 太多），但作为集成成员仍有价值（分辨率多样性）。
**集成方式对比（valid, TTA4，详见 compare_ensembles.py）**：等权概率平均 / 加权 / rank 平均 / logit 平均差异都在抽样噪声内；rank 平均会伤 AUROC，logit 平均 AUPRC 略高但 AUROC 略低。**选最稳健的等权概率平均**。

epoch 曲线（EMA AUROC, 384 线）：0.9714 → 0.9749 → 0.9821 → 0.9837 → 0.9843 → 0.9844（仍在涨，可继续训）

### 验证集最终评估（best.pt）
| 配置 | AUROC | AUPRC |
|---|---|---|
| 无 TTA（含年龄） | 0.98492 | 0.8047 |
| 无 TTA（年龄消融，用中位数代替） | 0.98434 | 0.8037 |
| **4x 翻转 TTA（含年龄）** | **0.98504** | **0.8159** |

结论：TTA 对小 AUROC 提升轻微但对 AUPRC 提升明显（+0.011）；年龄融合贡献 ~+0.0006 AUROC（图像已捕获大部分年龄信号）。

## 提交产物
- `submission.csv`（项目根目录）：5855 行，4x TTA，均值概率 0.0568，@0.5 阈值预测阳性 288 个（4.92%）
- `submit_pkg/`：`predict.py` + `model_weights.pt`(93MB) + README，已在本地用最终权重验证通过
- 权重备份：`D:\glaucoma_work\runs\efficientnet_b3_384_1014_194338_b3_384_cont\best.pt`，提交备份 `D:\glaucoma_work\submission_backup_v1.csv`

## 如何继续（给用户的操作手册）
```bash
# 环境（每次都先进入 venv）
VENV=/d/glaucoma_work/venv/Scripts/python.exe

# 1) 继续训练（热启动，更多 epoch 通常还能涨）
$VENV train.py --arch efficientnet_b3 --img-size 384 --epochs 5 \
  --batch-size 24 --accum 2 --workers 3 \
  --init-from "D:/glaucoma_work/runs/.../best.pt" --tag b3_384_more

# 2) 验证集评估（TTA）
$VENV evaluate.py --ckpt "<best.pt>" --tta 4

# 3) 生成提交
$VENV infer.py --ckpt "<best.pt>" --split test --tta 4 --out submission.csv

# 4) 换更大分辨率/更强骨干（注意显存，bs 要下调）
$VENV train.py --arch efficientnet_b4 --img-size 448 --batch-size 12 --accum 3 --workers 3

# 5) 多模型集成（先各自训练，再平均概率）
$VENV ensemble.py --ckpt runA/best.pt runB/best.pt --eval --tta 4          # 看验证集增益
$VENV ensemble.py --ckpt runA/best.pt runB/best.pt --weights 1 0.8 --submit --out submission.csv
```
**注意**：任何时候只跑一个 GPU 任务；训练用 `--workers 3`；崩溃后用 `--init-from` 恢复。

## ✅ Issue #3：train+valid 全量重训（工程侧已完成，2026-10-15）
完整手册见 **[docs/TRAIN_ON_ALL.md](docs/TRAIN_ON_ALL.md)**。要点：

- **原来 `--train-on-all` 根本跑不起来**：`CSV_ROOT`/`CACHE_ROOT` 写死队长机器路径，换机器必炸。已抽到 `paths.py`（环境变量 → 仓库布局 → 旧路径三级回退），`train.py` 另加 `--csv-root` / `--cache-root`
- **epoch 要按 step 对齐，不是按 epoch 对齐**：数据多 11.4%，`epochs_on_all = round(epochs/1.114)`。沿用 train-only 的 epoch 数就是多训 11.4% step —— B3@512 就是这么过拟合的
- **全量模型的 valid 指标是假的**，别回填；验收改用 `check_train_on_all.py`（与 train-only 参考模型在 test 上比 Spearman ≥0.90 + mean 相对偏差 ≤25%）
- **校准系数必须来自 train-only 模型**：全量模型在 valid 上概率虚高，用它拟合 shift 会把阈值推歪
- 修掉的配套缺陷：只写 `last.pt` 不写 `best.pt`（下游接不上）、无 val 仍建 val DataLoader（白占内存）、全量模式不写 `log.csv`（盲训）、下游靠 `-1 >= -1` 巧合选 EMA

## 待办 / 提升方向（按性价比排序）
- [ ] **继续训练**：epoch 曲线还在涨，再加 3-5 epoch 预计 +0.002~0.005
- [x] **train+valid 全量重训**：`--train-on-all` 工程侧就绪（Issue #3），待在有数据的机器上实跑
- [ ] **抬高分辨率微调**：B3@448 或 B4@384 微调 2-3 epoch，与现有模型做集成（两模型 T4 上 ~17ms/图，仍满足 100ms；总大小 150MB < 500MB）
- [ ] **复赛镜像内实测** predict.py（Python 3.12 + T4）
- [ ] 规则风险：仅允许主办方数据 → ImageNet 预训练权重属于通用初始化，行业惯例允许；如需绝对保守可 from scratch 训练（预计掉点明显，需 `pretrained=False` 支持）

## 已知坑（踩过的）
1. C 盘曾只剩 2.6GB：pip 临时目录/缓存必须指到 D 盘（TMP/TEMP/PIP_CACHE_DIR）
2. `py -3.11 -m venv` 的 ensurepip 在本机坏；用 `--without-pip` + get-pip.py
3. 本机页面文件 31.6GB、系统提交上限 48GB；torch 主进程吃 10-11GB、每个 worker 1.25GB → 多任务并发必崩
4. 线程预取（ThreadLoader）因 GIL 只有 22-25 张/秒，不要用（代码保留在 train.py 里备查）
5. channels_last 在本机慢 5 倍，别开
6. GPU 功率墙 70W（需管理员才能改），训练吞吐上限 ~55 张/秒（B3@384）

## 推理与部署
- `infer.py`：本地推理（读 cache512 + train.py 的模型定义），支持 4x 翻转 TTA，自动选 ema/model
- `submit_pkg/predict.py`：**复赛提交用的自包含脚本**（不依赖本地缓存/训练代码），已在本地验证能正确加载训练权重并输出 CSV
- `ensemble.py`：多模型概率平均（--eval 看验证集，--submit 出提交）
- `evaluate.py`：单 checkpoint 评估 + `--no-age` 消融

## 数据审计结论（已做）
- 标签：train 1657/51396=3.22%，valid 191/5849=3.27%，分布一致
- 年龄：阳性均值 62.8 岁 vs 阴性 56.5 岁；**年龄单特征 AUROC=0.695**（valid）→ 已做模态融合
- 年龄脏数据：train 有 0 岁、valid 有 137 岁 → 清洗：超出 [1,110] 用中位数 58 替换，再 clip [10,100]
- 图像：全是带黑边的彩色眼底图，原生尺寸 1725×1725 ~ 5472×3648 多种（多设备）
- **重复图：63100 个文件 md5 全部唯一，无任何精确重复**；dHash 的"高相似"是眼底图圆形视野导致的哈希碰撞（对照组同样相似），不是泄漏，无需特殊处理

## 预处理（cache512）
- 步骤：JPEG draft(768) 解码 → 灰度阈值>15 找眼底 bbox（0.2/99.8 分位去噪）→ 居中正方形 → BICUBIC 缩放到 512×512 → JPEG q92 存 D 盘
- 产物：`D:\glaucoma_work\cache512\{train,valid,test}\*.jpg` + `meta_all.csv` + 日志
- 速度：draft(768) 单张 ~59ms；8 进程 ~50 张/秒（16 进程反而变慢，磁盘争抢；与下载任务并发时会掉到 9 张/秒）
- 校验脚本：`verify_cache.py`（检查缺失/损坏/尺寸）

## 模型方案（train.py）
- 骨干：torchvision 预训练 EfficientNet-B3/B4（ImageNet 权重，非外部医学标注数据）+ **年龄 MLP 融合**（age → 16 维嵌入，与图像特征拼接 → 512 全连接 → logit）
- 不平衡：BCEWithLogitsLoss(pos_weight=5)；AUC 为排序指标，阈值无关
- 增强：RandomResizedCrop(0.7-1.0) + 水平/垂直翻转 + 旋转±25° + 颜色抖动 + 高斯模糊 + RandomErasing
- 训练：AdamW（骨干 1e-4 / 头 5e-4，wd 0.02）、fp16 AMP + GradScaler、cosine + 5% warmup、EMA(0.999)、grad clip 5.0
- 验证：每 epoch 在 valid 上算 AUROC/AUPRC（模型与 EMA 各算一次，取优）；保存 best.pt/last.pt 到 `D:\glaucoma_work\runs\<exp>\`
- 复赛部署：`--train-on-all` 用 train+valid 全量重训（无验证），配 `infer.py` 4x 翻转 TTA

## 推理（infer.py）
- `python infer.py --ckpt <best.pt> --split test --tta 4 --out submission.csv`
- 自动选 ema/model 中验证 AUROC 更高的权重；TTA 平均 4 个翻转的概率
- 输出列：Image_name（与原 CSV 逐字一致）、Probability（6 位小数）、Pred_Label（≥0.5）

## 待办 / 风险
- [ ] 冒烟测试（subset 跑通 train→infer→csv）
- [ ] 基线 B3@384 全量训练（估 ~10-15 min/epoch）
- [ ] 首版 submission.csv 尽早产出（每天限 2 次提交，先用掉 1 次探分）
- [ ] 提升方向：B4@448、train+valid 全量重训、双模型集成（T4 上两模型 ~80ms 仍满足 100ms、总大小 <500MB）
- [ ] 复赛模型打包：需兼容 torch 2.10 + Python 3.12（本地 3.11 的 checkpoint 通用，但打包代码要在 3.12 下测）
- [x] 规则风险：仅允许主办方数据 → ImageNet 预训练权重属于通用初始化而非外部标注医学数据，行业惯例允许；`--pretrained 0` 已支持 from-scratch 训练（#7）

## 🔒 from-scratch 合规备选（#7，P2 风险对冲）
- **现状**：ImageNet 预训练权重属于**通用初始化**（非外部标注医学数据），行业惯例允许，默认保留 `--pretrained 1`。
- **备选**：`python train.py --arch efficientnet_b3 --img-size 384 --pretrained 0 --tag scratch_probe` 可从头训练。
- **预期代价**（定量待 GPU 机器实测回填）：from-scratch 从头训练掉点明显（眼底图 + 3.2% 正类极不平衡 + 仅 5 万张），预计需 **2-3 倍 epoch** 且 AUROC 低 0.01~0.03。
- **触发条件**：仅在主办方**明确禁止**通用预训练权重时才启用；否则维持 pretrained。

## ⛔ RetFound 合规决策（2026-10-16）
- RetFound ViT-L（视网膜眼底数据预训练，属**外部医学数据**）**仅限内部消融参考，禁止进入任何提交文件 / 复赛包**。
- 两个理由：① 与"仅可用主办方数据"规则冲突（风险远高于 ImageNet 通用初始化）；② ViT-L 体积（fp32 ≈1.2GB / fp16 ≈0.6GB）超 500MB 复赛红线。
- 提交模型一律为 ConvNeXt / EfficientNet 系列。本决策已同步写入 `SUBMISSIONS_LOG.md` 提交流程规则与 `submit_pkg/README.md`。

## 协作（GitHub）
- 仓库（私有）：https://github.com/dongmingkun/glaucoma_first
- 后续工作已拆成工单（Issues），按优先级：
  - #1 [P0][评估] 判定官方评分口径 + 校准提交 A/B
  - #2 [P0][训练] B3@448 微调收尾 + 与 B3@384 集成
  - #3 [P1][训练] train+valid 全量重训（--train-on-all）→ **工程侧已就绪**（2026-10-15，手册 docs/TRAIN_ON_ALL.md），剩实跑
  - #4 [P1][训练] B4@384 / B4@448 对比实验
  - #5 [P0][工程] 复赛镜像实测（py3.12 + T4）+ predict.py 多模型支持
  - #6 [P2][流程] 提交台账与实验记录规范
  - #7 [P2][风险] from-scratch 合规备选路径
  - #8 [P2][探索] 建模探索池
- 规则：每天 2 次提交的任何使用先在对应 Issue 登记、提交后回填 LB 分数；实验结论（含失败）写回 Issue
- **重要更新（2026-10-14 深夜）**：
  - **#2 已完成**（队长机器上跑完）：B3@448 微调（AUPRC 0.8267）+ 384+448 集成（AUROC 0.98507 / AUPRC 0.82839）+ 校准版提交文件 `submission_20251015_ens_cal.csv` 已就绪 → **训练线队友请直接做 #3 / #4 / #8**
  - **#1 产物已就绪**：判别实验文件已生成，**无需等队员合并即可提交**（建议明天就交，锁定口径结论；队员合并后的产物留给后天）
  - **新增** `make_submission.py`（唯一提交入口，一条命令：评估→集成→校准→写CSV→格式校验→四口径预期分）
  - **#3 工程侧完成**（2026-10-15）：`paths.py` 路径解析 + `--train-on-all` 闭环修复 + `check_train_on_all.py` 验收 + `docs/TRAIN_ON_ALL.md` 手册。**训练线队友按手册实跑即可**，注意 epoch 要除以 1.114
  - **新增** `docs/TEAM_HANDOFF.md`（队员上手文档：环境/数据/命令/交付物约定/铁律）
- 分工（2026-10-14，待两位接受仓库邀请；如需调整在对应 Issue 回复）：
  - @Yulingling-sc：训练方向 → #2 448 集成、#3 全量重训、#4 B4 实验、#8 探索池
  - @JiahaoChen-star-JJ：评估/工程方向 → #1 评分口径、#5 复赛镜像、#6 提交台账、#7 合规备选

## 提交台账与实验记录规范（#6）
- 提交台账：`SUBMISSIONS_LOG.md`（列：日期/提交人/文件/md5/来源/TTA/校准/LB/结论）；提交前登记、提交后 10 分钟回填。
- 权重命名规范：`{arch}_{size}_{月日}_ep{epoch}_valid{auroc4位}.pt`，例 `b3_448_1015_ep3_valid0.9861.pt`；大文件不入库，走 Release 附件/群文件。
- 现有 run 清单（valid 指标，来源集成报告）：

| run | arch | size | AUROC | AUPRC |
|---|---|---|---|---|
| `convnext_tiny_384_1015_011832_cnxt_384` | convnext_tiny | 384 | 0.98945 | 0.85584 |
| `efficientnet_b3_384_1014_194338_b3_384_cont` | efficientnet_b3 | 384 | 0.98504 | 0.81592 |
| `efficientnet_b3_448_1014_221755_b3_448` | efficientnet_b3 | 448 | 0.98210 | 0.82672 |
| `efficientnet_b3_512_1014_233847_b3_512` | efficientnet_b3 | 512 | 0.98370 | 0.81415 |