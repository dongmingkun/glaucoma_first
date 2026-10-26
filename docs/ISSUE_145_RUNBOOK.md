# issue #145 Runbook — train+all Base@512 RA seed42（与 seed7 做 seed 平均，test-only 家族候选）

> 本文件是 #145 的**代码与执行手册预交付**。训练 / TTA4 推理需云端 24GB+ GPU；
> **seed 平均 + ens10 融合是 CPU**（`tools/seed_average.py`）。不占队长本地 5060。
> test-only 探索：train+all 无干净 valid，**不报 valid AUROC**，是否提分由一次线上提交回答；失败不降低 98.84。

## 背景
#138 的 train+all Base@512 RA（seed 7）线上实测 **98.84**（+0.02 vs 98.82），w10 融合首次真正提分；
但单 seed 的 test-only 证据仍薄。本单训 **seed 42**，与 seed7 做 logit 平均，降方差后再与 ens10 融合。

## 前置
- main 含 #131/#136/#140/#143/#144；
- `cache_all768`（train/valid 各 57245、test 5855，#134 已建）；
- `runs/cnxb_512_e16_ra/last.pt`（#128 train-only）做 check 对照；#138 的 seed7 test TTA4 在仓库根
  `cnxb_512_e16_ra_all_test_tta4.csv`（旧文档里的 `oof/` 是笔误；脚本仍兼容 oof/ 布局）。

## 一键（推荐）
```bash
CACHE=cache_all768 bash tools/pipeline_145.sh
```
脚本自动完成：git pull → 校验 cache_all768 计数 → train+all 16ep seed42 → test TTA4（`last.pt --use ema`）→ 5855/NaN 校验 → `check_train_on_all` vs #128 → seed 平均 + ens10 融合（w=0.10/0.15/0.20）→ seed42 直接融合（w=0.1/0.2/0.3）。

## 分步（与 #145 正文一致）
### 1) 训练（train+all，无干净 valid）
```bash
python train.py --arch convnext_base --img-size 512 --cache-root cache_all768 \
  --train-on-all --epochs 16 --batch-size 8 --accum 6 \
  --lr 1e-4 --head-lr 5e-4 --pretrained 1 --randaugment 2 9 --seed 42 \
  --loader workers --workers 8 --tag cnxb_512_e16_ra_all_s42 \
  --out-dir runs/cnxb_512_e16_ra_all_s42
```
- `last.pt/best.pt` 均为 EMA；**不要**用 valid 指标验收（valid 已并入训练）。

### 2) test TTA4 + check
```bash
python tools/oof_predict.py --ckpt runs/cnxb_512_e16_ra_all_s42/last.pt \
  --csv glaucoma_first/test.csv --cache-root cache_all768 --split-dir test \
  --img-size 512 --tta 4 --use ema \
  --out oof/cnxb_512_e16_ra_all_s42_test_tta4.csv

python check_train_on_all.py --ckpt runs/cnxb_512_e16_ra_all_s42/last.pt \
  --ref runs/cnxb_512_e16_ra/last.pt --tta 4 --cache-root cache_all768
```
- 报告与 #128 的 Spearman / 漂移 / top200；REVIEW/PASS 如实。

### 3) seed 平均 + ens10 融合（CPU）
```bash
# 纯 seed 平均：0.5*logit(seed7) + 0.5*logit(seed42)
python tools/seed_average.py \
  --members cnxb_512_e16_ra_all_test_tta4.csv oof/cnxb_512_e16_ra_all_s42_test_tta4.csv \
  --weights 0.5 0.5 --out oof/seed7_s42_test_tta4.csv

# seed 平均 + ens10 logit 融合（w=0.10/0.15/0.20，按 #145）
python tools/seed_average.py \
  --members cnxb_512_e16_ra_all_test_tta4.csv oof/cnxb_512_e16_ra_all_s42_test_tta4.csv \
  --weights 0.5 0.5 --fuse-ens10 submission_20251018_ens10_cal.csv \
  --fuse-weights 0.10 0.15 0.20 --fuse-mode logit --out-dir oof --tag seed7_s42

# seed42 直接与 ens10 融合（w=0.1/0.2/0.3，对照 seed7 的 98.84 w10）
python tools/fusion_ens10.py --ens10 submission_20251018_ens10_cal.csv \
  --new oof/cnxb_512_e16_ra_all_s42_test_tta4.csv --out-dir oof
```

## 验收（#145）
- test 5855 行、官方顺序、无 NaN、[0,1]；check 相关性合理；
- 明确 test-only：无干净 valid，最终是否提分由线上提交回答；**失败不降低 98.84**；
- 与 seed7 的 test Spearman（同配方换 seed）应较高（>0.95）；若明显漂移，seed 平均价值下降。

## 交付物
- `runs/cnxb_512_e16_ra_all_s42/last.pt`（Release/群文件）；
- `oof/cnxb_512_e16_ra_all_s42_test_tta4.csv`（5855）；
- `oof/seed7_s42_test_tta4.csv`（纯 seed 平均）+ `oof/seed7_s42_ens10_logit_w{10,15,20}.csv`；
- `oof/cnxb_512_e16_ra_all_s42_test_tta4_ens10_{logit,rank}_w{10,20,30}.csv`（seed42 直接融合）；
- `docs/ISSUE_RA_ALL_S42_RESULTS.md`：check/融合/与 seed7 一致性、失败原因（如有）。

## 时间
约 3–4h（4090 24GB）；排在 #139 之后（另一张 24GB 卡可并行）；不占队长本地 5060。

## 已做的 CPU 验证
- `python tools/seed_average.py --self-test`：合成 5855 行验证 logit 平均、对齐、权归归一、3 个融合候选生成——**PASS**；
- `py_compile`（seed_average.py）+ `bash -n`（pipeline_145.sh）通过。
