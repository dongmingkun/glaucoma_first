# #193 ConvNeXt-Large@384 + RA + train+all seed7（Kaggle T4×2）

> 状态：**代码侧已可执行**（补齐了工单依赖但仓库不存在的两个参数）；**训练本身待 Kaggle 会话执行**
> 本机无 GPU、无 Kaggle 凭据，训练不能在这里跑；本文交付的是"照着敲就能开跑"的完整流程 + 两个复核脚本。

## 0. 一句话结论

工单里给出的训练命令在修复前**跑不起来**：`train.py` 没有 `--resume-from` 和 `--save-every-steps`，argparse 会直接报
`unrecognized arguments` 退出。现已实现，并补了 `infer.py` 的两个路径覆盖参数，工单描述的"9h 会话 + 每 200 步 resume"策略因此才成立。

## 1. 改了什么（附验证方式）

| 文件 | 改动 | 为什么非改不可 | 验证 |
|---|---|---|---|
| `train.py` | 新增 `--save-every-steps N` | 工单正式训练命令带此参数，原仓库不存在 | 静态校验：工单用到的 19 个训练参数全部存在 |
| `train.py` | 新增 `--resume-from PATH` | 同上；没有它，8.5h timeout 之后只能从零重跑 | 同上 + 第 5 步的 3 分钟自测 |
| `train.py` | 原子写 `resume.pt`（写 `resume.tmp` → `os.replace`） | timeout 在保存瞬间打断会留下半个文件，下一会话直接读不出来 | 代码路径 |
| `train.py` | 捕获 `SIGTERM` / `SIGINT`，落盘后退出 | `timeout 30600` 发的就是 SIGTERM，默认处理会丢掉最后 200 步 | 代码路径 |
| `infer.py` | 新增 `--cache-root` / `--csv-root` | ckpt 里存的是训练时的缓存路径；下一会话把 Output 挂成 Input 后路径会变，推理必然找不到图 | 参数存在性 + `--help` |
| `tools/check_193_env.py` | 新增 | 开 GPU 前先把数据/显存/依赖查一遍，别拿计费时长试环境 | 本地实跑（缺数据时正确报 FAIL） |
| `tools/audit_193_seed7_csv.py` | 新增 | 交回来的 CSV 要过 10 项格式 + SHA256 + Spearman/top200 | 对 98.87 复现件实跑：10/10 PASS，md5 `4bf37b7c` 一致 |

## 2. 续跑语义（这里有个坑，工单没写）

余弦调度是 `total_steps = steps_per_epoch × --epochs`，靠 `global_step` 定位。如果只按"全局步数"接续、而 epoch 又从整数重开，剩余步数就和调度对不上；`lr_at()` 里 `p > 1` 时 `cos(pi·p)` 会**由 -1 回弹**，LR 不降反升。

所以 `resume.pt` 存的不是单一 `epoch`，而是三元组：

- `epoch_done`：已完整跑完的 epoch 数
- `it_done`：当前 epoch 已消费的 micro-batch 数
- `global_step`：优化步数（LR 位置的唯一依据）

恢复时从 `epoch_done + 1` 开始，并且**跳过该 epoch 前 `it_done` 个 batch**，剩余步数因此严格等于 `total_steps - global_step`。
代价：被跳过的 batch 仍会被 dataloader 解码（不解码就无法定位），按 200 步计在 T4 上约几分钟，可接受。

另外两个细节：

- `log.csv` 续跑时以追加模式打开；但新会话的 `/kaggle/working` 是干净的，日志会按会话分段，这是预期行为。
- `--epochs` 在续跑时必须**逐字不变**，它是调度总长的分母。

## 3. 开跑前必须先拍板的两条红线

### 3.1 封包（这条不过，25 小时白跑）

| 方案 | 体积 | 500MB 上限 |
|---|---|---|
| 现状 3 模型 | 475.82 MB | ✅ 余 24 MB |
| + Large fp16（约 350–390MB） | 826–868 MB | ❌ 超 ~70% |
| 用 Large 顶掉 cb384 | 约 650 MB | ❌ |
| 全 INT8（3+1 模型） | 视量化方案，需实测 | 可能压线 |

**结论：现有包体放不下第 4 个 Large 模型。** 训练前要先定"替换谁 / 是否全 INT8 / 放弃封包只留离线结论"。没定就不要开跑——这一点工单也写了，但没给算式。

### 3.2 数据合规

`linglingyusc/glaucoma` 必须是**私有** Dataset；眼底图属医疗数据，先确认赛题允许上传到 Kaggle，再传。

## 4. 时间预算（必须先测，不要按估值排）

- 每 epoch 优化步数：`(57245 // 4) // 12 = 1192`，16 epoch = 19,072 步
- 外推公式：`每 epoch 分钟 = 1192 × 实测 step 秒 / 60`
- 参照①：队长 Large@512 16ep 实跑 **1,138–1,183 s/epoch**（#167，同卡纯训练 5h12m）；Large@384 算力约为其 0.55 倍，同卡约 11–13 min/epoch
- 参照②：Base@512 是 630 s/epoch
- 但 T4 与上面两张卡都不同代，**别按比例硬缩**，以 smoke 实测为准（30h/周额度，16ep 若 40 min/epoch 就要 10.7h，正好两场会话）
- Kaggle T4×2 单卡 16GB，`train.py` 只用一张卡；30h/周额度

smoke 跑完 20 步后，用日志里的 `it21/…` 与秒数算出 step 时间，再决定 batch 要不要降档。

## 5. 开跑流程（Kaggle 上照抄）

### Step 0 · 环境预检（1 分钟，必做）

```bash
cd /kaggle/working && git clone https://github.com/dongmingkun/glaucoma_first.git
cd glaucoma_first && pip install -q timm scikit-learn pandas pillow
python tools/check_193_env.py --repo . --work-dir /kaggle/working
```

期望输出 `ALL CHECKS PASSED`，并给出三行数字：cache 布局（train=57245 或 51396+5849）、test=5855、训练样本缺图 0 张。
**缺图不等于 0 就是严重问题**：`MixedDS` 找不到图时会回退到 `cache/test/`，那等于拿测试图训练。

脚本末尾会打印按实际路径填好的训练命令，直接复制即可。

### Step 1 · 续跑自测（3 分钟，别跳过）

```bash
python train.py --arch convnext_base --img-size 256 --cache-root "$CACHE" --train-on-all \
  --epochs 2 --batch-size 4 --accum 2 --seed 7 --loader workers --workers 2 \
  --max-steps-per-epoch 5 --save-every-steps 3 \
  --out-dir /kaggle/working/runs/selftest --tag selftest

# 模拟断点：同一 out-dir，加 --resume-from
python train.py --arch convnext_base --img-size 256 --cache-root "$CACHE" --train-on-all \
  --epochs 2 --batch-size 4 --accum 2 --seed 7 --loader workers --workers 2 \
  --max-steps-per-epoch 5 --save-every-steps 3 --resume-from /kaggle/working/runs/selftest/resume.pt \
  --out-dir /kaggle/working/runs/selftest
```

第二条命令必须打印 `[resume] … 已完成 1/2 epoch，epoch 2 跳过 0 个 batch`，然后继续跑完第 2 个 epoch，
且 `log.csv` 里有 1、2 两行（追加而不是覆盖）。这一步不过，后面的 25 小时没有断点保护。

### Step 2 · smoke（先 batch 8，OOM 再降）

```bash
python train.py --arch convnext_large --img-size 384 --cache-root "$CACHE" \
  --train-on-all --epochs 1 --batch-size 8 --accum 6 \
  --lr 5e-5 --head-lr 2.5e-4 --pretrained 1 --randaugment 2 9 --seed 7 \
  --loader workers --workers 2 --max-steps-per-epoch 20 \
  --out-dir /kaggle/working/runs/smoke --tag smoke
```

OOM：`--batch-size 4 --accum 12`；仍 OOM：`2/24 --freeze-bn`（head-BN 风险已知）。
记下 `ep1 it20/…` 那行的秒数 → 算 `1192 × sec/20 / 60` = 每 epoch 分钟。

### Step 3 · 正式训练（8.5h，带 timeout 与断点）

```bash
timeout 30600 python train.py --arch convnext_large --img-size 384 --cache-root "$CACHE" \
  --train-on-all --epochs 16 --batch-size 4 --accum 12 \
  --lr 5e-5 --head-lr 2.5e-4 --pretrained 1 --randaugment 2 9 --seed 7 \
  --loader workers --workers 2 --save-every-steps 200 \
  --out-dir /kaggle/working/runs/cnxl_384_e16_ra_all_s7 --tag cnxl_384_e16_ra_all_s7
```

会话结束前点 **Save Version**，否则 `/kaggle/working` 里的 `resume.pt` 会丢。

### Step 4 · 下一会话续跑

```bash
RESUME=$(find /kaggle/input -name resume.pt | head -1)
mkdir -p /kaggle/working/runs/cnxl_384_e16_ra_all_s7
cp "$RESUME" /kaggle/working/runs/cnxl_384_e16_ra_all_s7/resume.pt
# 其余参数与 Step 3 逐字一致，只加一行：
#   --resume-from /kaggle/working/runs/cnxl_384_e16_ra_all_s7/resume.pt
```

### Step 5 · 推理 + 交付

```bash
python infer.py --ckpt /kaggle/working/runs/cnxl_384_e16_ra_all_s7/last.pt --split test \
  --tta 4 --workers 2 --batch-size 16 --use ema --cache-root "$CACHE" \
  --out /kaggle/working/oof/cnxl_384_e16_ra_all_s7_test_tta4.csv

python tools/audit_193_seed7_csv.py \
  --csv /kaggle/working/oof/cnxl_384_e16_ra_all_s7_test_tta4.csv \
  --ref seed7_s42=/kaggle/input/<seed7_s42 csv>/seed7_s42_test_tta4.csv \
  --json /kaggle/working/oof/audit_193_seed7.json
```

复核脚本查 10 项：5855 行 / 表头三列 / 官方名序 / 唯一名 / finite / [0,1] / 标签与概率一致 / CRLF / 无 BOM；
另给 SHA256、mean_prob、pos_rate，以及与传入 ref 的 Spearman / top200 / max_abs_diff。

**不要自动提交平台**，三人 CSV 汇总后统一评估。

## 6. 交付物

| 路径 | 内容 |
|---|---|
| `train.py` | `--resume-from` / `--save-every-steps` / 原子断点 / 信号处理 |
| `infer.py` | `--cache-root` / `--csv-root` 覆盖 |
| `tools/check_193_env.py` | 开跑前预检（CSV 行数、cache 布局与缺图、磁盘、GPU、依赖、现成命令） |
| `tools/audit_193_seed7_csv.py` | 交回 CSV 的格式/合规复核 |
| `docs/ISSUE_193_RESULTS.md` | 本文件 |

## 7. 缺口（如实）

1. **训练未执行**：任务本身是 GPU 训练，本机没有 GPU 也没有 Kaggle 凭据，这一步只能由持有 `linglingyusc` 账号的人在 Kaggle 上跑；
2. **续跑代码未在真 GPU 上实测**：本机无 torch，无法端到端跑通；Step 1 的 3 分钟自测就是为此设计的，跑之前先跑它；
3. **Kaggle 数据集未核验**：`cache_all` 的实际布局（57245 合并式 / 51396+5849 分离式）由 Step 0 的脚本判定，两种布局训练代码都支持；
4. **工单写的交付文档名是 `docs/ISSUE_192_S7_RESULTS.md`**（#192 已 closed，是父单），本单按实际编号建 `ISSUE_193_RESULTS.md`；
5. **封包红线未解**：见 3.1，需队长先定替换/量化方案。
