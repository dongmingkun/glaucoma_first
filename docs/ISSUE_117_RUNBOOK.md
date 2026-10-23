# issue #117 Runbook — ConvNeXt-Base@512/cache768 16ep 单模（CPU 预交付，云端 24GB+ 跑）

> 本文件是 #117 的**训练与 TTA4 推理流水线预交付**。实际训练 / 推理需云端 24GB+ GPU
> （3090/A5000/L20/L40 均可，不要求 4090），不占用队长本地 5060。
> 代码就绪后等云端资源一键跑通，再回填 `docs/ISSUE_117_RESULTS.md` 与 Release。

## 目标（#117）
- 从 ImageNet 初始化把 ConvNeXt-Base@512/cache768 训到 16ep（现有仅 4ep 热启动 0.99089、Base@384 8ep 0.99117）；
- valid TTA4 ≥ 0.9925，交付 valid/test 概率供集成与蒸馏；
- 单模，无 fold；不涉及 `--csv-root`，不改 `train.py`。

## 复现命令
### 一键（推荐）
```bash
# 在云端实例、干净的仓库 clone、仓库根目录、CUDA 可见时：
# 16ep 是长任务，必须 nohup/setsid，避免 SSH 断开被 SIGHUP 杀掉。
nohup env BS=4 ACCUM=12 CACHE=cache768 WORKERS=8 LOG=run117.log \
  bash tools/pipeline_117.sh > run117.out 2>&1 &
tail -f run117.log
# 24GB 卡更快：BS=8 ACCUM=6；16GB 卡：BS=2 ACCUM=24（等效 batch 48）
# CACHE 可指向任意含 train/valid/test 的 768px 缓存；test 在别的目录时加 TEST_CACHE=...
```
脚本自动完成：git pull → 校验 cache768 → 训练 16ep（ImageNet 初始化）→ valid/test TTA4（best 与 last 两口径）→ 行数/顺序/NaN 校验（5849/5855）。缓存目录缺失、train/valid 计数不是 51396/5849、已有旧 checkpoint、或任一推理/校验失败都会**立即非零退出**，不会打印 DONE。

### 单步（与 #117 正文一致）
**训练**（`--arch convnext_base --img-size 512 --cache-root cache768`）：
```bash
python train.py --arch convnext_base --img-size 512 --cache-root cache768 \
  --epochs 16 --batch-size 4 --accum 12 --lr 1e-4 --head-lr 5e-4 \
  --pretrained 1 --loader workers --workers 3 \
  --seed 42 --tag cnxb_512_e16
```
- 从 ImageNet 初始化（`--pretrained 1`，**禁止**用 valid-selected ckpt 热启动）；valid 保持干净（不要 train-on-all）；
- 24GB 卡可用 bs8×accum6；16GB 卡用 bs2×accum24；等效 batch 48；
- best.pt 按 train.py 默认的 blend（0.5×(AUROC+AUPRC)）选；同时报 last/EMA，避免只报一个口径；
- 云端 CPU 核多时可用 `WORKERS=8`（一键）或 `--workers 8`（单步）加速，不改变数据/超参；
- `cache768` 是 768px 源图缓存，训练/推理时 resize 到 512。

**valid / test TTA4 概率**（复用已合并的 `tools/oof_predict.py`，避免 infer.py 只能读全局 CSV 的限制）：
```bash
# valid（5849）
python tools/oof_predict.py --ckpt runs/cnxb_512_e16/best.pt \
  --csv glaucoma_first/valid.csv --cache-root cache768 --split-dir valid \
  --img-size 512 --tta 4 --out oof/cnxb_512_e16_valid_best_tta4.csv
# test（5855）
python tools/oof_predict.py --ckpt runs/cnxb_512_e16/best.pt \
  --csv glaucoma_first/test.csv --cache-root cache768 --split-dir test \
  --img-size 512 --tta 4 --out oof/cnxb_512_e16_test_best_tta4.csv
# last/EMA 口径：--ckpt last.pt --use ema 再跑一遍（best 用默认 auto）
```
- `best` 口径用 `--use auto`（train.py 选中的 best 权重）；`last` 口径**固定 `--use ema`**，对应 #117 的 “last/EMA”；

## 报告（真机侧，回填 `docs/ISSUE_117_RESULTS.md`）
- 每 epoch valid AUROC / EMA AUROC / AUPRC；
- valid TTA4 AUROC/AUPRC（**best 与 last/EMA 两口径**）；
- 与 Base@512 4ep、Base@384 8ep、ens9 的 **Spearman**；
- 与 ens9 的 **blend 增益**（w=0..0.5，配对 bootstrap）——可参考仓库 `compare_ensembles.py`；
- 若 valid TTA4 ≥0.9925 且与 ens9 相关 <0.95，标记为 package/蒸馏候选。

## 验收（真机指标，不在本预交付范围）
- 从 ImageNet 初始化（禁止 valid-selected 热启动）；
- valid/test CSV 行数 5849/5855、**Image_name 顺序与官方 CSV 完全一致**、概率在 [0,1]、无 NaN/inf；
- 报告 TTA4 与配对 bootstrap；
- 不占用队长本地 5060；云端 24GB+。

## 交付物清单（真机跑完回填）
- `runs/cnxb_512_e16/{best,last}.pt`（Release 或群文件）；
- `oof/cnxb_512_e16_valid_{best,last}_tta4.csv`（5849）+ `oof/cnxb_512_e16_test_{best,last}_tta4.csv`（5855）；
- `docs/ISSUE_117_RESULTS.md`：曲线、TTA4、相关性、增益。

## 备注（#117 正文）
- 如果 16ep 曲线在 ep12 后明显回落，按 best 交付，但同时报 last；
- 不要用 RetFound/外部医学权重；只 ImageNet 初始化；
- 与 #109/#118 的 fold 训练不同，本单**不需要** `cache_all`/`fold{k}_csvroot`。
