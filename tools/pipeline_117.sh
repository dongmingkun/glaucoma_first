#!/usr/bin/env bash
# #117 ConvNeXt-Base@512/cache768 16ep 单模训练 + valid/test TTA4 概率（云端 24GB+，后台 nohup 跑）
# 依赖（Phase A 校验）：
#   - 仓库 git pull 到含 PR #120（train.py --csv-root）+ PR #121（tools/oof_predict.py）
#   - cache768（512 图像缓存，train 51396 / valid 5849 / test 5855）在盘
# 设计依据：#117 正文 / 队长评论（03:52）。单模，无 fold；不涉及 --csv-root。
set -u
cd "$(dirname "$0")/.."   # 仓库根
PY="${PYTHON:-python}"
LOG="${LOG:-run117.log}"
CACHE="${CACHE:-cache768}"          # 512 图像缓存（train/valid/test 子目录）
TEST_CACHE="${TEST_CACHE:-$CACHE}"  # test 图像缓存（如单独存放可覆盖）
TAG="${TAG:-cnxb_512_e16}"
OUT="runs/${TAG}"
exec > >(tee -a "$LOG") 2>&1
echo "===== #117 pipeline start $(date) ====="

echo "=== [0] git pull ==="
git pull --ff-only || { echo "git pull 失败"; exit 1; }

echo "=== [1] 校验图像缓存 ==="
[ -d "$CACHE/train" ] && [ -d "$CACHE/valid" ] || { echo "ERROR: $CACHE/train|valid 不存在，需先重建缓存（preprocess.py）"; exit 1; }
[ -d "$TEST_CACHE/test" ] || echo "WARN: $TEST_CACHE/test 不存在，test TTA4 会失败；请用 TEST_CACHE=<含 test 的缓存> 覆盖"
T=$(ls "$CACHE/train" | wc -l); V=$(ls "$CACHE/valid" | wc -l)
echo "cache 就绪：train=$T valid=$V（期望 51396 / 5849）"
[ "$T" -eq 51396 ] && [ "$V" -eq 5849 ] || echo "WARN: train/valid 计数非 51396/5849，确认是否官方单模划分（单模训练不应含 valid）"

echo "=== [2] 训练 ConvNeXt-Base@512 16ep（ImageNet 初始化，禁止 valid-selected 热启动）==="
# 24GB 卡：bs8×accum6；16GB 卡：bs2×accum24；正文默认 bs4×accum12（等效 batch 48）
BS="${BS:-4}"; ACCUM="${ACCUM:-12}"
$PY train.py --arch convnext_base --img-size 512 --cache-root "$CACHE" \
  --epochs 16 --batch-size "$BS" --accum "$ACCUM" --lr 1e-4 --head-lr 5e-4 \
  --loader workers --workers 3 --seed 42 --tag "$TAG" --out-dir "$OUT" \
  || { echo "训练 FAILED rc=$?"; exit 1; }
[ -f "$OUT/best.pt" ] || { echo "ERROR: $OUT/best.pt 缺失"; exit 1; }
echo "训练完成：best.pt / last.pt -> $OUT"

echo "=== [3] valid / test TTA4 概率（best 与 last 两口径）==="
mkdir -p oof
for W in best last; do
  [ -f "$OUT/$W.pt" ] || { echo "ERROR: $OUT/$W.pt 缺失"; exit 1; }
  $PY tools/oof_predict.py --ckpt "$OUT/$W.pt" \
    --csv glaucoma_first/valid.csv --cache-root "$CACHE" --split-dir valid \
    --img-size 512 --tta 4 --out "oof/${TAG}_valid_${W}_tta4.csv" || { echo "valid($W) 预测 FAILED"; exit 1; }
  $PY tools/oof_predict.py --ckpt "$OUT/$W.pt" \
    --csv glaucoma_first/test.csv --cache-root "$TEST_CACHE" --split-dir test \
    --img-size 512 --tta 4 --out "oof/${TAG}_test_${W}_tta4.csv" || { echo "test($W) 预测 FAILED"; exit 1; }
done

echo "=== [4] 校验行数与 NaN ==="
$PY - "$TAG" <<'PYEOF'
import csv, sys, math
tag=sys.argv[1]
def check(p, want):
    n=0; bad=0
    with open(p,encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            n+=1
            try: v=float(r['Probability'])
            except: bad+=1; continue
            if math.isnan(v) or math.isinf(v): bad+=1
    print('%-42s rows=%d (期望 %d) nan/bad=%d'%(p,n,want,bad))
    return n==want and bad==0
ok=True
for w in ('best','last'):
    ok &= check('oof/%s_valid_%s_tta4.csv'%(tag,w),5849)
    ok &= check('oof/%s_test_%s_tta4.csv'%(tag,w),5855)
if not ok:
    print('ERROR: 行数或 NaN 校验失败'); sys.exit(1)
print('OK: valid 5849 / test 5855 行、无 NaN')
PYEOF

echo "=== [5] 报告（真机侧，回填 docs/ISSUE_117_RESULTS.md）==="
echo "下一步（可选，需 ens9/Base@512 4ep/Base@384 8ep 的 valid/test 概率 CSV）："
echo "  - valid TTA4 AUROC/AUPRC（best 与 last/EMA 两口径）"
echo "  - 与 ens9 / Base@512 4ep / Base@384 8ep 的 Spearman"
echo "  - 与 ens9 的 blend 增益（w=0..0.5，配对 bootstrap）——可参考仓库 compare_ensembles.py"
echo "产物：$OUT/{best,last}.pt + oof/${TAG}_{valid,test}_{best,last}_tta4.csv"
echo "===== #117 pipeline DONE $(date) ====="
