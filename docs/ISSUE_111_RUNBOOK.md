# issue #111 Runbook — 方案 A：3 折 Small 教师 + Small@512 学生集成蒸馏（CPU 预交付，云端 24GB+ / 本地 5060 跑）

> 本文件是 #111（方案 A）的**蒸馏流水线执行手册预交付**。`tools/gen_soft_labels.py` / `tools/distill.py`
> 代码已合并（PR #115），`train.py --csv-root` 修复（PR #120）、`tools/oof_predict.py`（PR #121）也已就位。
> 实际软标签生成 / 蒸馏 / 打包需 GPU（云端 24GB+ 或本地 5060 降级），不占队长本地 5060。
> 代码就绪后等资源一键跑通，再回填 `docs/ISSUE_111_RESULTS.md` 与 Release。

## 任务（#111 方案 A）
- **教师**：#109 的 3 折 ConvNeXt-Small OOF 模型（`runs/cnxs_fold{0,1,2}/last.pt`）；
- **学生**：ConvNeXt-Small@512，10ep；
- 目标：可部署单模型，≤500MB、T4 尽量 ≤100ms。

## 前置依赖
1. **先跑 #109**（`tools/pipeline_109.sh`）产出 3 个 fold `last.pt`；
2. `cache_all`（512 缓存，train/valid 各 57245，供 fold 教师软标签）、**`cache_all768`**（768 缓存，train/valid 各 57245，供 Small@512 学生 2B/test）、`cache768`、`cache512`；
3. 软标签尺度 / fold-OOF 教师 / 2A/2B 验收拆分等逻辑已在 `gen_soft_labels.py` / `distill.py`（PR #115）。

## 复现命令
### 一键（推荐）
```bash
# 在 GPU 实例、仓库根目录、CUDA 可见时（#109 已产出 3 折教师后）：
bash tools/pipeline_111.sh
```
脚本自动完成：git pull → 校验教师/缓存 → fold-OOF 软标签 → 2A 验收 → 2B 最终学生 → test 概率 → fp16 瘦身。

### 分步（与 #111 正文一致）
**Step 1 — fold-OOF 教师软标签**（train/valid 每行只取它所在 held-out 折的教师，test 用 3 教师平均）：
```bash
python tools/gen_soft_labels.py \
    --teacher runs/cnxs_fold0/last.pt --teacher runs/cnxs_fold1/last.pt --teacher runs/cnxs_fold2/last.pt \
    --fold-valid folds3/fold0_valid.csv folds3/fold1_valid.csv folds3/fold2_valid.csv \
    --fold-cache-root cache_all --cache-root cache512 \
    --split train valid test --tta 4 --out soft_fold_oof.csv
```

**Step 2A — 配方验收**（train-only + 干净 valid；验收学生 AUROC ≥ 教师 −0.001）：
```bash
python tools/distill.py --arch convnext_small --img-size 512 --cache-root cache768 \
    --epochs 10 --soft-labels soft_fold_oof.csv --kd-alpha 0.5 --kd-temp 2.0 \
    --lr 1e-4 --head-lr 5e-4 --pos-weight 5.0 --select-metric auroc \
    --out-dir runs/distill_cnxs512_eval
```
- 不加 `--train-on-all`；学生只在官方 train 训，valid 干净，用于验收。

**Step 2B — 最终学生**（train+valid，不用于验收）：
```bash
python tools/distill.py --arch convnext_small --img-size 512 --cache-root cache_all768 \
    --cache-size 768 --epochs 10 --soft-labels soft_fold_oof.csv \
    --kd-alpha 0.5 --kd-temp 2.0 --lr 1e-4 --head-lr 5e-4 --pos-weight 5.0 \
    --train-on-all --out-dir runs/distill_cnxs512_final
```
- **`cache_all768`** 从 `cache768` 用四行 `cp -rs` 合成（train/valid 各 57245，train+valid 全部图）；**不要用 512 的 `cache_all` 配 `--cache-size 768`**，否则训练缓存与打包预处理不一致；
- `--cache-size 768` 写进 checkpoint；`predict.py` 会按 768 预处理；
- train+valid 后 valid 不独立，不能做验收；最终学生按 2A 的配方/epoch 直接产出。

**Step 3 — test 概率 + fp16 瘦身**（pipeline 已含）：
```bash
python tools/oof_predict.py --ckpt runs/distill_cnxs512_final/best.pt \
  --csv glaucoma_first/test.csv --cache-root cache_all768 --split-dir test \
  --img-size 512 --tta 4 --out oof/distill_cnxs512_test_tta4.csv
python tools/slim_weights.py --in runs/distill_cnxs512_final/best.pt --out runs/distill_cnxs512_final/student_fp16.pt --use auto
```

## 教师优先级（按队长 #111 评论）
1. 首选 #109 的 3 折 Small OOF 模型当教师（每行软标签来自未见过它的模型，评估无污染）；
2. 时间紧则用 **ens9（train-only）** 起一版，等 fold 集成出来再替换；
3. ens10 的 train+all 家族**不能**用于 valid 软标签（会污染 valid）。

## 算力预估
- Small@512 10ep：4090/L40 约 **1.5–2.5h/次**，2A+2B 合计约 **3–5h**；
- 3090/A5000 约 **2.5–4h/次**，合计约 **5–8h**；
- 本地 5060 8GB 降级：2A 约 7h、2B 约 8h，合计 ~15h；学生 fp16 约 96MB。

## 验收（真机指标，不在本预交付范围）
- 2A 学生 valid AUROC ≥ 教师 −0.001；
- 包体 ≤500MB（Small 实际 ~96MB）；test 5855 行、顺序一致、无 NaN；
- 不占用队长本地 5060。

## 交付物清单（真机跑完回填）
- 学生 `runs/distill_cnxs512_final/{best,last}.pt` + `student_fp16.pt`（≤500MB）；
- `oof/distill_cnxs512_test_tta4.csv`（5855）；
- `docs/ISSUE_111_RESULTS.md`：教师/学生/基线、配方、延迟、包体；
- 打包目录（predict.py + 权重 + README）；T4 p95 未实测时按 #110 的单/双模型安全边际交付。
