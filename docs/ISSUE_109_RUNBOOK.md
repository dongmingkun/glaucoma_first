# issue #109 Runbook — 方案 A：3-fold ConvNeXt-Small@384 16ep（CPU 预交付，云端 24GB+ 跑）

> 本文件是 #109（方案 A）的**代码与执行手册预交付**。实际训练 / OOF / 打包需云端 24GB+ GPU
> （L20/L40/A6000/4090 最快，3090/A5000 亦可，16GB 也可 bs8×accum6），不占队长本地 5060。
> 代码就绪后等云端资源一键跑通，再回填 `docs/ISSUE_109_RESULTS.md` 与 Release。

## 任务（#109 方案 A）
- 不跑原 5-fold Base（保留在 `folds/` 作备选），改为 **3-fold ConvNeXt-Small@384 16ep**（`folds3/`，PR #119 已合并）。
- 目标：拿 1848 正例的 OOF + 3 个数据子集不同的模型；Small 在云端比 Base 快，适合当前时间预算。
- **OOF 用 `last.pt` + 固定 `--use ema`**（`best.pt` 按 held-out valid 选会高估 OOF；
  `--use auto` 按该折 AUROC 选 EMA/model 同样是选择泄漏，见 #126 组长评论）；valid/test 都 TTA4；test 用 3 折平均。

## 关键修复（PR #120，已合并）
`train.py --csv-root` 原来只改 `train.py` 的全局名，`paths.load_rows()` 仍读默认全量 → fold 会**静默训成全量**。
已改为 `load_rows(split, csv_root=args.csv_root)`，验证 fold0 为 38163/19082 行（正例 1232/616）。
本 runbook 走 `--csv-root folds3/fold{k}_csvroot`，**禁用 `--extra-rows`**（fold CSV 列名 `True_Label`，`--extra-rows` 读 `Label` 会 KeyError）。

## 复现命令
### 一键（推荐）
```bash
# 在云端实例、仓库根目录、CUDA 可见时：
BS=8 ACCUM=6 CACHE_SRC=cache512 LOG=run109.log bash tools/pipeline_109.sh
```
脚本自动完成：git pull → 建 `fold{k}_csvroot` → 合并 `cache_all`（train/valid 各 57245，符号链接不占盘）
→ 3 折 Small@384 训练 → `last.pt` 出 OOF + test → 组装 `oof/oof_probs.npy`（57245）+ `oof/test_probs_tta4.csv`（5855）。

### 单折（调试 / 续跑，与 #109 正文一致）
```bash
for k in 0 1 2; do
  python train.py --arch convnext_small --img-size 384 --cache-root cache_all \
    --csv-root folds3/fold${k}_csvroot --epochs 16 --batch-size 8 --accum 6 \
    --lr 1e-4 --head-lr 5e-4 --loader workers --workers 3 --seed 42 --tag cnxs_fold${k}
done
```
- 每折从 ImageNet 初始化；**不要** `--init-from` 指向 valid-selected ckpt。

### OOF / test（单折，复用已合并的 tools/oof_predict.py；固定 --use ema）
```bash
# held-out valid（OOF 必须用 last.pt + 固定 --use ema，禁止 auto/best）
python tools/oof_predict.py --ckpt runs/cnxs_fold0/last.pt \
  --csv folds3/fold0_csvroot/valid.csv --cache-root cache_all --split-dir valid \
  --img-size 384 --tta 4 --use ema --out oof/cnxs_fold0_valid_tta4.csv
# test（与 OOF 同口径，供 #127 stacking 配对）
python tools/oof_predict.py --ckpt runs/cnxs_fold0/last.pt \
  --csv glaucoma_first/test.csv --cache-root cache_all --split-dir test \
  --img-size 384 --tta 4 --use ema --out oof/cnxs_fold0_test_tta4.csv
```

## 缓存（cache_all 建法，见 folds3/README.md）
- `cache_all/train` 与 `cache_all/valid` 都放全部 57245 张（`cp -rs` 符号链接，不占额外磁盘）；
- 优先复用现有 cache512/cache768；新实例用 `preprocess.py` 重建（1–2h）。

## 算力预估（云端 24GB+）
- L20/L40/A6000/4090：Small@384 16ep 约 **2.5–4h/折**，3 折约 **8–12h**；
- 3090/A5000：约 **4–6h/折**，3 折约 **12–18h**。

## 验收（真机指标，不在本预交付范围）
- 3 折 valid 并集 = 全部 57245，无 train/valid 重叠；
- OOF 用 `last.pt` + 固定 `--use ema`；test 5855 行、顺序一致、无 NaN；
- 3 折模型两两 Spearman < 0.98；
- 与 ens9 的配对 bootstrap（OOF vs ens9 在 valid 子集上的对比）。

## 交付物清单（真机跑完回填）
- 3 个 `runs/cnxs_fold{k}/last.pt`（Release/群文件）；
- `oof/oof_probs.npy`（57245）+ `oof/test_probs_tta4.csv`（5855，3 折平均）；
- `docs/ISSUE_109_RESULTS.md`：OOF AUROC/AUPRC、每折、3 折 Spearman、与 ens9 的配对 bootstrap、复现命令。

## 下游依赖
- **#111（集成蒸馏，方案 A）** 用本单的 3 个 fold `last.pt` 当 OOF 教师：`tools/gen_soft_labels.py --teacher runs/cnxs_fold{0,1,2}/last.pt ...`，见 `docs/ISSUE_111_PLAN.md`。
