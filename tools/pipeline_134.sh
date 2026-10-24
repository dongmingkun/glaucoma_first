#!/usr/bin/env bash
# #134 RA Base@512 16ep × 3-fold clean OOF（#128 最强新成员的 fold 版，#127 99 门槛前置）
# 依赖（Phase A 校验）：
#   - main（含 PR #131：OOF 固定 --use ema；tools/oof_report.py 含 --names-only）
#   - folds3/ + fold{k}_csvroot（pipeline 会自建 csvroot）
#   - cache768（768px 缓存，训练 resize 512；train 51396 / valid 5849 / test 5855）在盘
# 设计依据：#134 正文（训练/OOF 命令与验收）。泄漏红线：OOF 一律 last.pt + 固定 --use ema；
#   禁止 best.pt / --use auto / --train-on-all；每折 ImageNet 初始化。
set -u
cd "$(dirname "$0")/.."   # 仓库根
PY="${PYTHON:-python}"
LOG="${LOG:-run134.log}"
CACHE_SRC="${CACHE_SRC:-cache768}"   # 原始 cache768（train 51396 / valid 5849 / test 5855）
FOLDS=folds3
SEED="${SEED:-7}"          # #134 正文：seed 7（与 #128 RA 单模一致）
RA="${RA:-2 9}"            # #134 正文：RandAugment 2 9
BS="${BS:-8}"; ACCUM="${ACCUM:-6}"; WORKERS="${WORKERS:-8}"   # 24GB 卡
exec > >(tee -a "$LOG") 2>&1
echo "===== #134 pipeline start $(date) ====="

echo "=== [0] git pull ==="
git pull --ff-only || { echo "git pull 失败"; exit 1; }

echo "=== [1] 校验图像缓存 ==="
[ -d "$CACHE_SRC/train" ] && [ -d "$CACHE_SRC/valid" ] && [ -d "$CACHE_SRC/test" ] \
  || { echo "ERROR: $CACHE_SRC/train|valid|test 不存在，需先重建缓存（preprocess.py）"; exit 1; }
T0=$(ls "$CACHE_SRC/train" | wc -l); V0=$(ls "$CACHE_SRC/valid" | wc -l); S0=$(ls "$CACHE_SRC/test" | wc -l)
echo "cache768 就绪：train=$T0 valid=$V0 test=$S0（期望 51396/5849/5855）"
[ "$T0" -eq 51396 ] && [ "$V0" -eq 5849 ] && [ "$S0" -eq 5855 ] \
  || { echo "ERROR: cache768 计数 $T0/$V0/$S0 != 51396/5849/5855"; exit 1; }

echo "=== [1.5] 建 folds3/fold{k}_csvroot ==="
for k in 0 1 2; do
  mkdir -p "$FOLDS/fold${k}_csvroot"
  cp "$FOLDS/fold${k}_train.csv" "$FOLDS/fold${k}_csvroot/train.csv"
  cp "$FOLDS/fold${k}_valid.csv" "$FOLDS/fold${k}_csvroot/valid.csv"
done

echo "=== [2] 合并 cache_all768（train/valid 各 57245 + test 5855，符号链接不占盘）==="
rm -rf cache_all768 && mkdir -p cache_all768/train cache_all768/valid cache_all768/test
cp -rs "$CACHE_SRC/train/." cache_all768/train/
cp -rs "$CACHE_SRC/valid/." cache_all768/train/
cp -rs "$CACHE_SRC/train/." cache_all768/valid/
cp -rs "$CACHE_SRC/valid/." cache_all768/valid/
cp -rs "$CACHE_SRC/test/." cache_all768/test/
T=$(ls cache_all768/train | wc -l); V=$(ls cache_all768/valid | wc -l); S=$(ls cache_all768/test | wc -l)
echo "cache_all768 就绪：train=$T valid=$V test=$S"
[ "$T" -eq 57245 ] && [ "$V" -eq 57245 ] && [ "$S" -eq 5855 ] \
  || { echo "ERROR: cache_all768 计数 $T/$V/$S != 57245/57245/5855"; exit 1; }

echo "=== [3] 3 折训练（ConvNeXt-Base@512 16ep + RA $RA，seed $SEED，ImageNet 初始化）==="
for k in 0 1 2; do
  [ ! -e "runs/ra_fold${k}/last.pt" ] \
    || { echo "ERROR: runs/ra_fold${k} 已有 checkpoint；换 OUT 或先归档（不自动覆盖）"; exit 1; }
  echo "----- ra_fold$k ($(date)) -----"
  # RA 故意不加引号以按空格分词（与 pipeline_117.sh 一致）
  $PY train.py --arch convnext_base --img-size 512 --cache-root cache_all768 \
    --csv-root "$FOLDS/fold${k}_csvroot" --epochs 16 --batch-size "$BS" --accum "$ACCUM" \
    --lr 1e-4 --head-lr 5e-4 --randaugment $RA --seed "$SEED" \
    --loader workers --workers "$WORKERS" \
    --tag ra_fold${k} --out-dir runs/ra_fold${k} \
    || { echo "ra_fold $k FAILED rc=$?"; exit 1; }
  echo "ra_fold$k done"
done

echo "=== [4] OOF（held-out valid）+ test 概率（一律 last.pt + 固定 --use ema）==="
mkdir -p oof
for k in 0 1 2; do
  [ -f "runs/ra_fold${k}/last.pt" ] || { echo "ra_fold $k last.pt 缺失"; exit 1; }
  $PY tools/oof_predict.py --ckpt "runs/ra_fold${k}/last.pt" \
    --csv "$FOLDS/fold${k}_csvroot/valid.csv" --cache-root cache_all768 --split-dir valid \
    --img-size 512 --tta 4 --use ema --out "oof/ra_fold${k}_valid_tta4.csv" \
    || { echo "ra_fold $k valid 预测 FAILED"; exit 1; }
  $PY tools/oof_predict.py --ckpt "runs/ra_fold${k}/last.pt" \
    --csv glaucoma_first/test.csv --cache-root cache_all768 --split-dir test \
    --img-size 512 --tta 4 --use ema --out "oof/ra_fold${k}_test_tta4.csv" \
    || { echo "ra_fold $k test 预测 FAILED"; exit 1; }
done

echo "=== [5] 组装 OOF（官方 train+valid 顺序）+ test 3 折平均 ==="
$PY - <<'PYEOF'
import csv, numpy as np, os, sys
order=[]
for fn in ('glaucoma_first/train.csv','glaucoma_first/valid.csv'):
    with open(fn,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f): order.append(r['Image_name'])
idx={n:i for i,n in enumerate(order)}
oof=np.zeros(len(order)); seen=set()
for k in range(3):
    p='oof/ra_fold%d_valid_tta4.csv'%k
    if not os.path.exists(p): continue
    with open(p,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            i=idx[r['Image_name']]; oof[i]=float(r['Probability']); seen.add(i)
if len(seen) != len(order):
    print('ERROR: OOF 覆盖 %d/%d 不完整，停止'%(len(seen),len(order))); sys.exit(1)
np.save('oof/ra_oof.npy', oof)
print('OOF cover=%d/%d saved oof/ra_oof.npy'%(len(seen),len(order)))
trows=[]
with open('glaucoma_first/test.csv',encoding='utf-8-sig',newline='') as f:
    trows=[r['Image_name'] for r in csv.DictReader(f)]
tp=np.zeros(len(trows)); cnt=0
for k in range(3):
    p='oof/ra_fold%d_test_tta4.csv'%k
    if not os.path.exists(p): continue
    d={}
    with open(p,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f): d[r['Image_name']]=float(r['Probability'])
    tp+=[d[n] for n in trows]; cnt+=1
if cnt != 3:
    print('ERROR: test folds=%d != 3'%cnt); sys.exit(1)
if len(tp) != 5855:
    print('ERROR: test rows=%d != 5855'%len(tp)); sys.exit(1)
with open('oof/ra_test_tta4.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['Image_name','Probability','Pred_Label'])
    for n,v in zip(trows,tp): w.writerow([n,'%.6f'%v,1 if v>=0.5 else 0])
print('test rows',len(tp),'test_mean',round(float(tp.mean()),5))
PYEOF
rc=$?
[ "$rc" -eq 0 ] || { echo "ERROR: 组装失败 rc=$rc"; exit 1; }

echo "=== [6] 报告提示（真机侧回填 docs/ISSUE_RA_3FOLD_RESULTS.md）==="
echo "  每折/OOF 指标：oof_report.py metrics --labels folds3/fold{k}_valid.csv --probs oof/ra_fold{k}_valid_tta4.csv"
echo "  OOF 57245：oof_report.py metrics --labels glaucoma_first/train.csv glaucoma_first/valid.csv --probs oof/ra_oof.npy"
echo "  折间 Spearman：oof_report.py metrics --names-only --labels glaucoma_first/test.csv --probs oof/ra_fold{0,1,2}_test_tta4.csv"
echo "产物：runs/ra_fold{0,1,2}/last.pt + oof/ra_fold{k}_{valid,test}_tta4.csv + oof/ra_oof.npy + oof/ra_test_tta4.csv"
echo "===== #134 pipeline DONE $(date) ====="
