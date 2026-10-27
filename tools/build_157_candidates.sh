#!/usr/bin/env bash
# #157：Swin-B@384 train+all seed123 → test TTA4 → ens10 融合 + RA_avg/Swin_pair rank 组合 + 校验
# 前置：runs/swinb_384_all_s123/last.pt 已存在（train_157.sh 产出）、cache_all768、以下组件 CSV 在仓库目录：
#   submission_20251018_ens10_cal.csv（ens10）
#   cnxb_512_e16_ra_test_best_tta4.csv / _last_tta4.csv（RA_avg）
#   swinb_384_e16_test_probs_tta4.csv / swinb_384_all_v3_test_probs_tta4.csv（Swin_pair）
#   oof/seed7_s42_test_tta4.csv（seedavg，仅用于重建 98.87 对照）
set -u
set -o pipefail
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python}"
LOG="${LOG:-run157b.log}"
exec > >(tee -a "$LOG") 2>&1
echo "===== #157 candidates start $(date) ====="

ENS10=submission_20251018_ens10_cal.csv
RA="cnxb_512_e16_ra_test_best_tta4.csv,cnxb_512_e16_ra_test_last_tta4.csv"
SWP="swinb_384_e16_test_probs_tta4.csv,swinb_384_all_v3_test_probs_tta4.csv"
SEEDAVG=oof/seed7_s42_test_tta4.csv
NEW=oof/swinb_384_all_s123_test_tta4.csv

for f in "$ENS10" cnxb_512_e16_ra_test_best_tta4.csv cnxb_512_e16_ra_test_last_tta4.csv \
         swinb_384_e16_test_probs_tta4.csv swinb_384_all_v3_test_probs_tta4.csv "$NEW"; do
  [ -f "$f" ] || { echo "ERROR: 缺组件 $f"; exit 1; }
done

echo "=== [1] ens10 + new Swin（logit，w=0.10/0.20/0.30）==="
$PY tools/fusion_ens10.py --ens10 "$ENS10" --new "$NEW" --out-dir oof --weights 0.10 0.20 0.30 \
  || { echo "fusion_ens10 FAILED"; exit 1; }

echo "=== [2] 重建 98.87 配方作对照（0.60 ens10 + 0.10 seedavg + 0.15 RA_avg + 0.15 Swin_pair）==="
$PY tools/rank_blend.py \
  --member "ens10=0.60:$ENS10" \
  --member "seedavg=0.10:$SEEDAVG" \
  --member "ra_avg=0.15:$RA" \
  --member "swin_pair=0.15:$SWP" \
  --out oof/blend157_ctl_987.csv \
  || { echo "control blend FAILED"; exit 1; }

echo "=== [3] #157 主候选 B1 / B2（rank）==="
$PY tools/rank_blend.py \
  --member "ens10=0.50:$ENS10" --member "ra_avg=0.20:$RA" \
  --member "swin_pair=0.15:$SWP" --member "swin_s123=0.15:$NEW" \
  --ref oof/blend157_ctl_987.csv --out oof/blend157_b1_w50201515.csv \
  || echo "WARN B1 失败"
$PY tools/rank_blend.py \
  --member "ens10=0.55:$ENS10" --member "ra_avg=0.20:$RA" \
  --member "swin_pair=0.10:$SWP" --member "swin_s123=0.15:$NEW" \
  --ref oof/blend157_ctl_987.csv --out oof/blend157_b2_w55201015.csv \
  || echo "WARN B2 失败"

echo "=== [3b] 更强 Swin 暴露备选 B3（0.45 ens10 + 0.20 RA + 0.10 Swin_pair + 0.25 new）==="
$PY tools/rank_blend.py \
  --member "ens10=0.45:$ENS10" --member "ra_avg=0.20:$RA" \
  --member "swin_pair=0.10:$SWP" --member "swin_s123=0.25:$NEW" \
  --ref oof/blend157_ctl_987.csv --out oof/blend157_b3_w45201025.csv \
  || echo "WARN B3 失败"

echo "=== [4] 事件：v3 与 new 的 seed soup（若 fp16 ckpt 可读则做）==="
if $PY - <<'PY'
import sys
try:
    import torch
    ck = torch.load('/root/autodl-tmp/swinb_384_all_v3_fp16.pt', map_location='cpu', weights_only=False)
    ok = isinstance(ck, dict) and 'ema' in ck
    print('v3 fp16 ckpt keys:', list(ck)[:6] if isinstance(ck, dict) else type(ck), 'has_ema=', ok)
    sys.exit(0 if ok else 3)
except Exception as e:
    print('read fail:', type(e).__name__, str(e)[:120]); sys.exit(3)
PY
then
  cp /root/autodl-tmp/swinb_384_all_v3_fp16.pt ./_swinb_v3_fp16.pt
  $PY tools/seed_soup.py --ckpts ./_swinb_v3_fp16.pt runs/swinb_384_all_s123/last.pt \
    --weights 0.5 0.5 --out runs/swinb_384_v3_s123_soup/last.pt \
    && $PY tools/oof_predict.py --ckpt runs/swinb_384_v3_s123_soup/last.pt \
         --csv glaucoma_first/test.csv --cache-root cache_all768 --split-dir test \
         --img-size 384 --tta 4 --use ema --out oof/swinb_384_v3_s123_soup_test_tta4.csv \
    && $PY tools/rank_blend.py \
         --member "ens10=0.50:$ENS10" --member "ra_avg=0.20:$RA" \
         --member "swin_pair=0.15:$SWP" --member "swin_soup=0.15:oof/swinb_384_v3_s123_soup_test_tta4.csv" \
         --ref oof/blend157_ctl_987.csv --out oof/blend157_b4_soup_w50201515.csv \
    || echo "WARN soup 分支失败（跳过，不影响主候选）"
else
  echo "SKIP: v3 fp16 无 ema 键或不可读 → 按工单「不在就跳过」"
fi

echo "=== [5] 强制校验 ==="
$PY - <<'PY'
import csv, glob, math, os
def rd(p):
    with open(p, encoding='utf-8', newline='') as f:
        return [(x['Image_name'], float(x['Probability'])) for x in csv.DictReader(f)]
def rank(v):
    n=len(v); o=sorted(range(n),key=lambda i:v[i]); r=[0.0]*n; i=0
    while i<n:
        j=i
        while j+1<n and v[o[j+1]]==v[o[i]]: j+=1
        a=(i+j)/2+1
        for k in range(i,j+1): r[o[k]]=a
        i=j+1
    return r
def pear(x,y):
    n=len(x); mx=sum(x)/n; my=sum(y)/n
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
def sp(a,b): return pear(rank(a),rank(b))
def top(ns,ps,k=200): return set(ns[i] for i in sorted(range(len(ps)),key=lambda i:-ps[i])[:k])

official=[x['Image_name'] for x in csv.DictReader(open('glaucoma_first/test.csv',encoding='utf-8'))]
NEW='oof/swinb_384_all_s123_test_tta4.csv'
refs={'seed7':'cnxb_512_e16_ra_all_test_tta4.csv',
      'seed42':'oof/cnxb_512_e16_ra_all_s42_test_tta4.csv',
      's123':'oof/cnxb_512_e16_ra_all_s123_test_tta4.csv'}
print('\n--- new Swin 基础校验 + 与 RA 三 seed 的相关 ---')
rows=rd(NEW); ns=[n for n,_ in rows]; ps=[p for _,p in rows]
bad=sum(1 for v in ps if v!=v or v<0 or v>1)
print('newSwin rows=%d order=%s bad=%d mean=%.5f pos@.5=%d range=[%.4f,%.4f]'
      %(len(rows), ns==official, bad, sum(ps)/len(ps), sum(1 for v in ps if v>=0.5), min(ps), max(ps)))
tn=top(ns,ps)
for k,v in refs.items():
    if not os.path.exists(v): print('  vs %-7s 缺失，跳过'%k); continue
    rr=rd(v); m=dict(rr); rv=[m[n] for n in ns]; tr=top(ns,rv)
    print('  vs %-7s Spearman=%.4f top200=%d'%(k, sp(ps,rv), len(tn&tr)))

print('\n--- 候选 vs 重建的 98.87 对照 ---')
ctl=rd('oof/blend157_ctl_987.csv'); cn=[n for n,_ in ctl]; cp=[p for _,p in ctl]; ct=top(cn,cp)
cands=['oof/swinb_384_all_s123_test_tta4_ens10_logit_w10.csv',
       'oof/swinb_384_all_s123_test_tta4_ens10_logit_w20.csv',
       'oof/swinb_384_all_s123_test_tta4_ens10_logit_w30.csv',
       'oof/blend157_b1_w50201515.csv','oof/blend157_b2_w55201015.csv',
       'oof/blend157_b3_w45201025.csv','oof/blend157_b4_soup_w50201515.csv']
ok=True
for p in cands:
    if not os.path.exists(p): continue
    rr=rd(p); m=dict(rr); ns2=[n for n,_ in rr]; v=[m[n] for n in cn]
    b=sum(1 for x in v if x!=x or x<0 or x>1)
    good=(len(rr)==5855) and (ns2==official) and b==0
    ok = ok and good
    print('  %-48s rows=%d order=%s bad=%d mean=%.4f pos=%d rho_ctl=%.4f top200=%d %s'
          %(os.path.basename(p), len(rr), ns2==official, b, sum(v)/len(v),
            sum(1 for x in v if x>=0.5), sp(cp,v), len(ct&top(ns2,v)), 'OK' if good else 'FAIL'))
print('RESULT:', 'PASS' if ok else 'FAIL')
PY

echo "===== #157 candidates DONE $(date) ====="
ls -l oof/swinb_384_all_s123_test_tta4*.csv oof/blend157_*.csv 2>/dev/null
