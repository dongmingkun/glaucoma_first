#!/usr/bin/env bash
# #109 方案 A：3-fold ConvNeXt-Small@384 16ep 训练 + OOF/test 概率（云端 24GB+，后台 nohup 跑）
# 依赖（Phase A 校验）：
#   - 仓库 git pull 到含 PR #119（folds3/）+ PR #120（train.py --csv-root 修复）+ PR #121（tools/oof_predict.py）
#   - cache512（384 图像缓存，train+valid 各 57245 张、test 5855 张）在盘
# 设计依据：#109 正文「方案 A」/ 队长评论（03:58 / 04:00）。OOF 用 last.pt（不要 best.pt）。
set -u
cd "$(dirname "$0")/.."   # 仓库根
PY="${PYTHON:-python}"
LOG="${LOG:-run109.log}"
CACHE_SRC="${CACHE_SRC:-cache512}"   # 384 图像缓存（jpg），train/valid 各 57245
FOLDS=folds3                         # PR #119 已合并的分层 3 折（方案 A 复用 folds3）
exec > >(tee -a "$LOG") 2>&1
echo "===== #109 pipeline start $(date) ====="

echo "=== [0] git pull ==="
git pull --ff-only || { echo "git pull 失败"; exit 1; }

echo "=== [1] 校验图像缓存 ==="
if [ ! -d "$CACHE_SRC/train" ] || [ ! -d "$CACHE_SRC/valid" ]; then
  echo "ERROR: $CACHE_SRC/train|valid 不存在，需先重建缓存（preprocess.py）"; exit 1
fi

echo "=== [1.5] 建 folds3/fold{k}_csvroot（复制 fold CSV，列名 Image_name,Age,True_Label）==="
for k in 0 1 2; do
  mkdir -p "$FOLDS/fold${k}_csvroot"
  cp "$FOLDS/fold${k}_train.csv" "$FOLDS/fold${k}_csvroot/train.csv"
  cp "$FOLDS/fold${k}_valid.csv" "$FOLDS/fold${k}_csvroot/valid.csv"
done

echo "=== [2] 合并 cache_all（train+valid 全部 57245 图，符号链接不占盘）==="
rm -rf cache_all && mkdir -p cache_all/train cache_all/valid cache_all/test
cp -rs "$CACHE_SRC/train/." cache_all/train/
cp -rs "$CACHE_SRC/valid/." cache_all/train/
cp -rs "$CACHE_SRC/train/." cache_all/valid/
cp -rs "$CACHE_SRC/valid/." cache_all/valid/
[ -d "$CACHE_SRC/test" ] && cp -rs "$CACHE_SRC/test/." cache_all/test/
echo "cache_all 就绪：train=$(ls cache_all/train | wc -l) valid=$(ls cache_all/valid | wc -l)"

echo "=== [3] 3 折训练（ConvNeXt-Small@384 16ep，ImageNet 初始化）==="
# 24GB 卡：bs8×accum6；16GB 卡亦可 bs8×accum6（#109 正文）。每折 ImageNet 初始化，勿用 --init-from 指向 valid-selected ckpt。
BS="${BS:-8}"; ACCUM="${ACCUM:-6}"
for k in 0 1 2; do
  echo "----- fold $k ($(date)) -----"
  $PY train.py --arch convnext_small --img-size 384 --cache-root cache_all \
    --csv-root "$FOLDS/fold${k}_csvroot" --epochs 16 --batch-size "$BS" --accum "$ACCUM" \
    --lr 1e-4 --head-lr 5e-4 --loader workers --workers 3 --seed 42 \
    --tag cnxs_fold${k} --out-dir runs/cnxs_fold${k} \
    && echo "fold $k done" || echo "fold $k FAILED rc=$?"
done

echo "=== [4] OOF（held-out valid）+ test 概率（一律 last.pt）==="
mkdir -p oof
for k in 0 1 2; do
  [ -f "runs/cnxs_fold${k}/last.pt" ] || { echo "fold $k last.pt 缺失，跳过"; continue; }
  $PY tools/oof_predict.py --ckpt "runs/cnxs_fold${k}/last.pt" \
    --csv "$FOLDS/fold${k}_csvroot/valid.csv" --cache-root cache_all --split-dir valid \
    --img-size 384 --tta 4 --out "oof/cnxs_fold${k}_valid_tta4.csv"
  $PY tools/oof_predict.py --ckpt "runs/cnxs_fold${k}/last.pt" \
    --csv glaucoma_first/test.csv --cache-root cache_all --split-dir test \
    --img-size 384 --tta 4 --out "oof/cnxs_fold${k}_test_tta4.csv"
done

echo "=== [5] 组装 OOF（按 train+valid 顺序）+ test 3 折平均 ==="
$PY - <<'PYEOF'
import csv, numpy as np, os
order=[]
for fn in ('glaucoma_first/train.csv','glaucoma_first/valid.csv'):
    with open(fn,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f): order.append(r['Image_name'])
idx={n:i for i,n in enumerate(order)}
oof=np.zeros(len(order)); seen=set()
for k in range(3):
    p='oof/cnxs_fold%d_valid_tta4.csv'%k
    if not os.path.exists(p): continue
    with open(p,encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            i=idx[r['Image_name']]; oof[i]=float(r['Probability']); seen.add(i)
np.save('oof/oof_probs.npy', oof)
print('OOF cover=%d/%d (期望 57245)'%(len(seen),len(order)))
trows=[]
with open('glaucoma_first/test.csv',encoding='utf-8-sig',newline='') as f:
    trows=[r['Image_name'] for r in csv.DictReader(f)]
tp=np.zeros(len(trows)); cnt=0
for k in range(3):
    p='oof/cnxs_fold%d_test_tta4.csv'%k
    if not os.path.exists(p): continue
    d={}
    with open(p,encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f): d[r['Image_name']]=float(r['Probability'])
    tp+=[d[n] for n in trows]; cnt+=1
tp/=max(cnt,1)
with open('oof/test_probs_tta4.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['Image_name','Probability','Pred_Label'])
    for n,v in zip(trows,tp): w.writerow([n,'%.6f'%v,1 if v>=0.5 else 0])
print('test rows',len(tp),'test_mean',round(float(tp.mean()),5))
PYEOF
echo "===== #109 pipeline DONE $(date) ====="
