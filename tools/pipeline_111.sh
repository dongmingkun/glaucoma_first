#!/usr/bin/env bash
# #111 方案 A 集成蒸馏流水线（CPU 预交付；真机需云端 24GB+ 或本地 5060，后台 nohup 跑）
# 依赖（Phase A 校验）：
#   - 仓库 git pull 到含 PR #115（gen_soft_labels/distill）+ PR #120（train.py --csv-root）+ PR #121（oof_predict）
#   - #109 已产出 3 个 fold 教师：runs/cnxs_fold{0,1,2}/last.pt
#   - cache_all（train/valid 各 57245，符号链接）、cache768（512px 学生用）、cache512（test）在盘
# 设计依据：#111 正文「方案 A」/ 队长评论（03:58 / 04:00）。
set -u
cd "$(dirname "$0")/.."   # 仓库根
PY="${PYTHON:-python}"
LOG="${LOG:-run111.log}"
TEACHERS="${TEACHERS:-runs/cnxs_fold0/last.pt runs/cnxs_fold1/last.pt runs/cnxs_fold2/last.pt}"
exec > >(tee -a "$LOG") 2>&1
echo "===== #111 pipeline start $(date) ====="

echo "=== [0] git pull ==="
git pull --ff-only || { echo "git pull 失败"; exit 1; }

echo "=== [1] 校验前置：#109 教师 + 缓存 ==="
for t in $TEACHERS; do [ -f "$t" ] || { echo "ERROR: 教师 $t 缺失（先跑 #109 pipeline_109.sh）"; exit 1; }; done
for d in cache_all cache768 cache512; do [ -d "$d" ] || { echo "ERROR: $d 缺失，按 folds3/README.md / preprocess.py 重建"; exit 1; }; done
[ -d cache_all/train ] && echo "cache_all/train=$(ls cache_all/train | wc -l)（期望 57245）"

echo "=== [1.5] 合并 cache_all768（cache768 train+valid 全部 57245，供 Small@512 学生）==="
[ -d cache768/test ] || { echo "ERROR: cache768/test 缺失（学生测试要用 768 缓存）"; exit 1; }
rm -rf cache_all768 && mkdir -p cache_all768/train cache_all768/valid cache_all768/test
cp -rs cache768/train/. cache_all768/train/
cp -rs cache768/valid/. cache_all768/train/
cp -rs cache768/train/. cache_all768/valid/
cp -rs cache768/valid/. cache_all768/valid/
cp -rs cache768/test/. cache_all768/test/
T768=$(ls cache_all768/train | wc -l); V768=$(ls cache_all768/valid | wc -l)
echo "cache_all768 就绪：train=$T768 valid=$V768"
[ "$T768" -eq 57245 ] && [ "$V768" -eq 57245 ] || { echo "ERROR: cache_all768 计数 $T768/$V768"; exit 1; }

echo "=== [2] 生成 fold-OOF 教师软标签（train/valid 取 held-out 折，test 取 3 教师平均）==="
TEACHER_ARGS=""
for t in $TEACHERS; do TEACHER_ARGS="$TEACHER_ARGS --teacher $t"; done
$PY tools/gen_soft_labels.py $TEACHER_ARGS \
  --fold-valid folds3/fold0_valid.csv folds3/fold1_valid.csv folds3/fold2_valid.csv \
  --fold-cache-root cache_all --cache-root cache512 \
  --split train valid test --tta 4 --out soft_fold_oof.csv \
  || { echo "soft labels FAILED"; exit 1; }
[ -s soft_fold_oof.csv ] || { echo "ERROR: soft_fold_oof.csv 为空"; exit 1; }
echo "soft labels -> soft_fold_oof.csv ($(wc -l < soft_fold_oof.csv) 行)"

echo "=== [3] 2A 配方验收（train-only + 干净 valid，学生 AUROC ≥ 教师 −0.001）==="
$PY tools/distill.py --arch convnext_small --img-size 512 --cache-root cache768 \
  --epochs 10 --soft-labels soft_fold_oof.csv --kd-alpha 0.5 --kd-temp 2.0 \
  --lr 1e-4 --head-lr 5e-4 --pos-weight 5.0 --select-metric auroc \
  --out-dir runs/distill_cnxs512_eval --tag cnxs512_kd_eval \
  || { echo "2A FAILED rc=$?"; exit 1; }
[ -f runs/distill_cnxs512_eval/best.pt ] || { echo "ERROR: 2A best.pt 缺失"; exit 1; }
echo "2A done"

echo "=== [4] 2B 最终学生（train+valid，不用于验收）==="
$PY tools/distill.py --arch convnext_small --img-size 512 --cache-root cache_all768 \
  --cache-size 768 --epochs 10 --soft-labels soft_fold_oof.csv \
  --kd-alpha 0.5 --kd-temp 2.0 --lr 1e-4 --head-lr 5e-4 --pos-weight 5.0 \
  --train-on-all --out-dir runs/distill_cnxs512_final --tag cnxs512_kd_final \
  || { echo "2B FAILED rc=$?"; exit 1; }
[ -f runs/distill_cnxs512_final/best.pt ] || { echo "ERROR: 2B best.pt 缺失"; exit 1; }
echo "2B done"

echo "=== [5] 最终学生 test 概率（TTA4）==="
mkdir -p oof
STU=runs/distill_cnxs512_final/best.pt
[ -f "$STU" ] || STU=runs/distill_cnxs512_final/last.pt
$PY tools/oof_predict.py --ckpt "$STU" \
  --csv glaucoma_first/test.csv --cache-root cache_all768 --split-dir test \
  --img-size 512 --tta 4 --out oof/distill_cnxs512_test_tta4.csv

echo "=== [6] fp16 瘦身（≤500MB，Small 实际约 96MB）+ 打包提示 ==="
$PY tools/slim_weights.py --in "$STU" --out runs/distill_cnxs512_final/student_fp16.pt --use auto \
  || { echo "slim FAILED"; exit 1; }
[ -f runs/distill_cnxs512_final/student_fp16.pt ] || { echo "ERROR: student_fp16.pt 缺失"; exit 1; }
echo "交付物：runs/distill_cnxs512_final/{best,last}.pt + student_fp16.pt + oof/distill_cnxs512_test_tta4.csv"
echo "打包目录（predict.py + 权重 + README）见 #110；T4 p95 未实测按 #110 单/双模型安全边际交付。"
echo "===== #111 pipeline DONE $(date) ====="
