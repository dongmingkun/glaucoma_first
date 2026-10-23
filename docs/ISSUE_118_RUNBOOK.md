# issue #118 Runbook — Swin-B@384 16ep × 3-fold（CPU 预交付，云端 24GB+ 跑）

> 本文件是 #118 的**代码与执行手册预交付**（与 #111 的 `docs/ISSUE_111_PLAN.md` 同模式）。
> 实际 Swin-B 训练 / OOF / 打包需要云端 24GB+ GPU（3090/A5000/L20/L40 均可，不要求 4090），
> 不占用队长本地 5060。代码就绪后等云端资源一键跑通，再回填 `docs/ISSUE_118_RESULTS.md` 与 Release。

## 任务回顾（#118）
- 复用 `folds/make_folds.py --n-folds 3` 的 3 折（已合并 PR #119：`folds3/`，57245 行 / 1848 正例，分层均衡、无重叠）。
- 每折从 ImageNet 初始化训 Swin-B@384 16ep；OOF 用 `last.pt`（不要 `best.pt`）。
- 交付：OOF（57245 行）、每折指标、3 折 Spearman、与 ens9 / Base fold 的 Spearman；test 用 3 折平均（5855 行）。

## 关键修复（PR #120，已合并）
`train.py --csv-root` 原来只改 `train.py` 的全局名，`paths.load_rows()` 仍读默认全量——
3 折会**静默训成全量**。已改为 `load_rows(split, csv_root=args.csv_root)`，验证 fold0 为
38163/19082 行（正例 1232/616）。本 runbook 的 `pipeline_118.sh` 直接走 `--csv-root folds3/fold{k}_csvroot`，
**禁用 `--extra-rows`**（fold CSV 列名 `True_Label`，`--extra-rows` 读 `Label` 会 KeyError）。

## 复现命令
### 一键（推荐）
```bash
# 在云端实例、仓库根目录、CUDA 可见时：
BS=8 ACCUM=6 CACHE_SRC=cache512 LOG=run118.log \
  bash tools/pipeline_118.sh
# 16GB 卡改为：BS=4 ACCUM=12
```
脚本自动完成：git pull → 建 `fold{k}_csvroot` → 合并 `cache_all`（train/valid 各 57245，符号链接不占盘）
→ 3 折训练 → `last.pt` 出 OOF + test → 组装 `oof/oof_probs.npy`（57245）+ `oof/test_probs_tta4.csv`（5855）。

### 单折（调试 / 续跑）
```bash
python train.py --arch swin_base --img-size 384 --cache-root cache_all \
  --csv-root folds3/fold0_csvroot --epochs 16 --batch-size 8 --accum 6 \
  --lr 1e-4 --head-lr 5e-4 --loader workers --workers 3 --seed 42 --tag swin_fold0
```

### OOF / test（单折，复用 tools/oof_predict.py）
```bash
# held-out valid（OOF 必须用 last.pt）
python tools/oof_predict.py --ckpt runs/swin_fold0/last.pt \
  --csv folds3/fold0_csvroot/valid.csv --cache-root cache_all --split-dir valid \
  --img-size 384 --tta 4 --out oof/swin_fold0_valid_tta4.csv
# test
python tools/oof_predict.py --ckpt runs/swin_fold0/last.pt \
  --csv glaucoma_first/test.csv --cache-root cache_all --split-dir test \
  --img-size 384 --tta 4 --out oof/swin_fold0_test_tta4.csv
```

## 为什么需要 `tools/oof_predict.py`
`infer.py` 只读全局 `glaucoma_first/{train,valid,test}.csv`，无法对每折 held-out valid 预测，
会导致 OOF 泄漏。本脚本接受任意 `--csv` + `--cache-root/--split-dir`，做 TTA 预测；
同时被 #109 / #111 的 OOF 复用。

## 验收（真机指标，不在本预交付范围）
- 3 折 OOF 覆盖全部 57245、无 train/valid 重叠；
- OOF 用 `last.pt`；test 5855 行、顺序一致、无 NaN；
- 不占用队长本地 5060；云端 24GB+。

## 交付物清单（真机跑完回填）
- `folds3/` CSV + 3 个 `runs/swin_fold{k}/last.pt`；
- `oof/oof_probs.npy`（57245）+ `oof/test_probs_tta4.csv`（5855）；
- `docs/ISSUE_118_RESULTS.md`：OOF AUROC/AUPRC、每折、3 折 Spearman、与 ens9/Base fold 的 Spearman、配对 bootstrap。
