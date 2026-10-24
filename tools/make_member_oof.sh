#!/usr/bin/env bash
# #127 生成单个 train-only 成员的 3 折 held-out OOF（57245，官方顺序）+ test TTA4 概率。
# 用法：
#   NAME=cb384 CKPT=runs/cb384/best.pt CACHE=cache_all IMG=384 bash tools/make_member_oof.sh
# 说明：
#   - 学生/成员按 folds3/fold{k}_csvroot/valid.csv 逐折预测（每行只由没见过它的模型预测）；
#   - @384 成员用 CACHE=cache_all（由 cache512 四行 cp -rs 合成，含全部 57245）；
#   - @512 成员用 CACHE=cache_all768（由 cache768 合成，方法同 folds3/README.md）；
#   - 不要给 train+all 成员生成 OOF（它们看过 valid，会泄漏）；
#   - OOF 一律用 held-out 折的模型（last.pt/对应折权重），不要用 valid-selected best.pt。
set -u
cd "$(dirname "$0")/.."   # 仓库根
PY="${PYTHON:-python}"
NAME="${NAME:?需要 NAME（成员名，如 cb384）}"
CKPT="${CKPT:?需要 CKPT（该成员的训练权重）}"
CACHE="${CACHE:-cache_all}"
TEST_CACHE="${TEST_CACHE:-$CACHE}"
IMG="${IMG:-384}"
TTA="${TTA:-4}"
USE="${USE:-auto}"
FOLDS="${FOLDS:-folds3}"
mkdir -p oof
echo "===== make_member_oof: $NAME (ckpt=$CKPT cache=$CACHE img=$IMG tta=$TTA use=$USE) ====="
for k in 0 1 2; do
  [ -f "$FOLDS/fold${k}_csvroot/valid.csv" ] || { echo "ERROR: $FOLDS/fold${k}_csvroot/valid.csv 缺失"; exit 1; }
  $PY tools/oof_predict.py --ckpt "$CKPT" --csv "$FOLDS/fold${k}_csvroot/valid.csv" \
    --cache-root "$CACHE" --split-dir valid --img-size "$IMG" --tta "$TTA" --use "$USE" \
    --out "oof/${NAME}_fold${k}_valid_tta4.csv" || { echo "fold $k 失败"; exit 1; }
done
$PY - "$NAME" <<'PYEOF'
import csv, sys, numpy as np
name = sys.argv[1]
order = []
for fn in ('glaucoma_first/train.csv', 'glaucoma_first/valid.csv'):
    with open(fn, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            order.append(r['Image_name'])
idx = {n: i for i, n in enumerate(order)}
oof = np.full(len(order), np.nan, dtype=np.float64)
for k in range(3):
    p = 'oof/%s_fold%d_valid_tta4.csv' % (name, k)
    with open(p, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            oof[idx[r['Image_name']]] = float(r['Probability'])
bad = int(np.isnan(oof).sum())
if bad:
    print('ERROR: OOF 有 %d 行未覆盖（folds3 未覆盖全量？）' % bad)
    sys.exit(1)
np.save('oof/%s_oof.npy' % name, oof)
print('wrote oof/%s_oof.npy (%d)' % (name, len(oof)))
PYEOF
$PY tools/oof_predict.py --ckpt "$CKPT" --csv glaucoma_first/test.csv \
  --cache-root "$TEST_CACHE" --split-dir test --img-size "$IMG" --tta "$TTA" --use "$USE" \
  --out "oof/${NAME}_test_tta4.csv" || { echo "test 预测失败"; exit 1; }
echo "done: oof/${NAME}_oof.npy + oof/${NAME}_test_tta4.csv"
