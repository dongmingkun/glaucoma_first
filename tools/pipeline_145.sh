#!/usr/bin/env bash
# #145 train+all Base@512 RA seed42 + test TTA4 + check + seed 平均 + ens10 融合候选（云端 24GB+）
# 设计依据：#145 正文 / #138 结果（seed7 test-only 98.84）。train+all 无干净 valid，不报 valid AUROC。
# 依赖：
#   - main 含 #131/#136/#140/#143/#144（fusion_ens10/check_train_on_all/seed_average 等）
#   - cache_all768（train/valid 各 57245、test 5855；#134 已建）
#   - runs/cnxb_512_e16_ra/last.pt（#128 train-only，作 check 对照）；#138 seed7 的 test TTA4 作 seed 平均输入
set -u
set -o pipefail
cd "$(dirname "$0")/.."   # 仓库根
PY="${PYTHON:-python}"
LOG="${LOG:-run145.log}"
CACHE="${CACHE:-cache_all768}"
TAG="${TAG:-cnxb_512_e16_ra_all_s42}"
OUT="runs/${TAG}"
REF="${REF:-runs/cnxb_512_e16_ra/last.pt}"                           # #128 train-only
SEED7="${SEED7:-oof/cnxb_512_e16_ra_all_test_tta4.csv}"              # #138 seed7 test TTA4
ENS10="${ENS10:-submission_20251018_ens10_cal.csv}"                  # 98.82 基线
exec > >(tee -a "$LOG") 2>&1
echo "===== #145 pipeline start $(date) ====="

echo "=== [0] git pull ==="
git pull --ff-only || { echo "git pull 失败"; exit 1; }

echo "=== [1] 校验 cache_all768（train/valid 各 57245、test 5855）==="
[ -d "$CACHE/train" ] && [ -d "$CACHE/valid" ] && [ -d "$CACHE/test" ] \
  || { echo "ERROR: $CACHE/{train,valid,test} 缺失；先跑 tools/pipeline_134.sh 建 cache_all768"; exit 1; }
T=$(ls "$CACHE/train" | wc -l | tr -d ' '); V=$(ls "$CACHE/valid" | wc -l | tr -d ' '); S=$(ls "$CACHE/test" | wc -l | tr -d ' ')
echo "cache_all768：train=$T valid=$V test=$S（期望 57245/57245/5855）"
[ "$T" -eq 57245 ] && [ "$V" -eq 57245 ] && [ "$S" -eq 5855 ] \
  || { echo "ERROR: cache_all768 计数 $T/$V/$S != 57245/57245/5855"; exit 1; }

echo "=== [2] 训练 train+all Base@512 RA seed42（16ep）==="
[ ! -e "$OUT/last.pt" ] && [ ! -e "$OUT/best.pt" ] \
  || { echo "ERROR: $OUT 已有 checkpoint；换 TAG 或先归档（不自动覆盖）"; exit 1; }
$PY train.py --arch convnext_base --img-size 512 --cache-root "$CACHE" \
  --train-on-all --epochs 16 --batch-size 8 --accum 6 \
  --lr 1e-4 --head-lr 5e-4 --pretrained 1 --randaugment 2 9 --seed 42 \
  --loader workers --workers 8 --tag "$TAG" --out-dir "$OUT" \
  || { echo "训练 FAILED rc=$?"; exit 1; }
[ -f "$OUT/last.pt" ] || { echo "ERROR: $OUT/last.pt 缺失"; exit 1; }
echo "训练完成：last.pt/best.pt（均为 EMA）-> $OUT"

echo "=== [3] test TTA4（last.pt，--use ema）==="
mkdir -p oof
TEST_OUT="oof/${TAG}_test_tta4.csv"
$PY tools/oof_predict.py --ckpt "$OUT/last.pt" \
  --csv glaucoma_first/test.csv --cache-root "$CACHE" --split-dir test \
  --img-size 512 --tta 4 --use ema --out "$TEST_OUT" \
  || { echo "test 预测 FAILED"; exit 1; }
$PY - "$TEST_OUT" <<'PYEOF'
import csv, sys, math
p = sys.argv[1]; n = 0; bad = 0; lo = 1.0; hi = 0.0
with open(p, encoding='utf-8', newline='') as f:
    for r in csv.DictReader(f):
        n += 1
        v = float(r['Probability']); lo = min(lo, v); hi = max(hi, v)
        if math.isnan(v) or math.isinf(v) or v < 0 or v > 1: bad += 1
print('test rows=%d (期望 5855) 越界/NaN=%d range=[%.4f,%.4f]' % (n, bad, lo, hi))
if n != 5855 or bad: sys.exit(1)
PYEOF

echo "=== [4] check_train_on_all vs #128 train-only ==="
if [ -f "$REF" ]; then
  $PY check_train_on_all.py --ckpt "$OUT/last.pt" --ref "$REF" --tta 4 --cache-root "$CACHE" \
    || echo "WARN: check_train_on_all 返回非零（见上）"
else
  echo "WARN: 对照 $REF 不存在，跳过 check（只做 test 分布自检）"
  $PY check_train_on_all.py --ckpt "$OUT/last.pt" --tta 4 --cache-root "$CACHE" || true
fi

echo "=== [5] seed 平均（seed7+seed42）+ ens10 融合（w=0.10/0.15/0.20）==="
if [ -f "$SEED7" ] && [ -f "$ENS10" ]; then
  $PY tools/seed_average.py \
    --members "$SEED7" "$TEST_OUT" --weights 0.5 0.5 \
    --out "oof/seed7_s42_test_tta4.csv" \
    --fuse-ens10 "$ENS10" --fuse-weights 0.10 0.15 0.20 --fuse-mode logit \
    --out-dir oof --tag seed7_s42 \
    || echo "WARN: seed 平均/融合失败（见上）"
else
  [ -f "$SEED7" ] || echo "WARN: seed7 $SEED7 缺失，跳过 seed 平均（需 #138 产出）"
  [ -f "$ENS10" ] || echo "WARN: ens10 $ENS10 缺失，跳过融合"
fi

echo "=== [6] seed42 直接与 ens10 融合（w=0.1/0.2/0.3，对照 seed7 的 98.84 w10）==="
if [ -f "$ENS10" ]; then
  $PY tools/fusion_ens10.py --ens10 "$ENS10" --new "$TEST_OUT" --out-dir oof \
    || echo "WARN: fusion_ens10 失败（见上）"
fi

echo "=== 交付物 ==="
echo "  $OUT/{last,best}.pt"
echo "  $TEST_OUT"
echo "  oof/seed7_s42_test_tta4.csv（纯 seed 平均）"
echo "  oof/seed7_s42_ens10_logit_w{10,15,20}.csv（seed 平均 + ens10）"
echo "  oof/${TAG}_test_tta4_ens10_{logit,rank}_w{10,20,30}.csv（seed42 直接融合）"
echo "  -> 回填 docs/ISSUE_RA_ALL_S42_RESULTS.md（check/融合/与 seed7 一致性；test-only，失败不降低 98.84）"
echo "===== #145 pipeline DONE $(date) ====="
