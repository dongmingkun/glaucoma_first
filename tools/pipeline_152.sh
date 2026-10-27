#!/usr/bin/env bash
# #152：train+all Base@512 RA seed123（16ep）→ test TTA4 → 3-seed 概率平均 / 3-seed soup → ens10 融合 → 强制校验
# 算力：4090 24GB（训练约 3h）；全程 fail-fast，任何一步失败即停并打印原因。
# 设计依据：#152 正文（2026-10-26）+ 队长评论（16:19Z「关键路径只有这条 seed123」）。
# 依赖：cache_all768（57245/57245/5855）；seed7 = runs/cnxb_512_e16_ra_all/last.pt；
#       seed42 = runs/cnxb_512_e16_ra_all_s42/last.pt；tools/{seed_average,seed_soup,oof_predict,fusion_ens10}.py
set -u
set -o pipefail
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python}"
LOG="${LOG:-run152.log}"
CACHE="${CACHE:-cache_all768}"
TAG="${TAG:-cnxb_512_e16_ra_all_s123}"
OUT="runs/${TAG}"
S7="${S7:-runs/cnxb_512_e16_ra_all/last.pt}"
S42="${S42:-runs/cnxb_512_e16_ra_all_s42/last.pt}"
SOUP3="${SOUP3:-runs/cnxb_512_e16_ra_all_soup3/last.pt}"
ENS10="${ENS10:-submission_20251018_ens10_cal.csv}"
WEIGHTS="${WEIGHTS:-0.10 0.15 0.20 0.25}"
exec > >(tee -a "$LOG") 2>&1
echo "===== #152 pipeline start $(date) ====="

echo "=== [0] git pull（失败不阻塞）==="
git pull --ff-only 2>&1 | tail -2 || echo "WARN: git pull 跳过（浅克隆/本地 remote）"

echo "=== [1] cache_all768 校验（期望 57245/57245/5855）==="
[ -d "$CACHE/train" ] && [ -d "$CACHE/valid" ] && [ -d "$CACHE/test" ] \
  || { echo "ERROR: $CACHE/{train,valid,test} 缺失"; exit 1; }
T=$(ls "$CACHE/train" | wc -l | tr -d ' '); V=$(ls "$CACHE/valid" | wc -l | tr -d ' '); S=$(ls "$CACHE/test" | wc -l | tr -d ' ')
echo "cache_all768: train=$T valid=$V test=$S"
[ "$T" -eq 57245 ] && [ "$V" -eq 57245 ] && [ "$S" -eq 5855 ] \
  || { echo "ERROR: $CACHE 计数 $T/$V/$S != 57245/57245/5855"; exit 1; }
[ -f "$S7" ]  || { echo "ERROR: seed7 $S7 缺失（Release ra_all-1025 → aria2c 下载）"; exit 1; }
[ -f "$S42" ] || { echo "ERROR: seed42 $S42 缺失（Release ra_all_s42）"; exit 1; }

echo "=== [2] 训练 seed123（16ep，train+all）==="
if [ -f "$OUT/last.pt" ]; then
  echo "已存在 $OUT/last.pt，跳过训练"
else
  $PY train.py --arch convnext_base --img-size 512 --cache-root "$CACHE" \
    --train-on-all --epochs 16 --batch-size 8 --accum 6 \
    --lr 1e-4 --head-lr 5e-4 --pretrained 1 --randaugment 2 9 --seed 123 \
    --loader workers --workers 8 --tag "$TAG" --out-dir "$OUT" \
    || { echo "训练 FAILED rc=$?"; exit 1; }
fi
[ -f "$OUT/last.pt" ] || { echo "ERROR: $OUT/last.pt 缺失"; exit 1; }
echo "训练完成：$(ls -l "$OUT/last.pt" | awk '{print $5}') B -> $OUT"

echo "=== [3] seed123 test TTA4 ==="
mkdir -p oof
S123_CSV="oof/${TAG}_test_tta4.csv"
$PY tools/oof_predict.py --ckpt "$OUT/last.pt" \
  --csv glaucoma_first/test.csv --cache-root "$CACHE" --split-dir test \
  --img-size 512 --tta 4 --use ema --out "$S123_CSV" \
  || { echo "PREDICT FAILED rc=$?"; exit 1; }

echo "=== [4] 3-seed 概率平均（seed7=根目录 CSV；seed42/s123=oof/ 下产物）==="
AVG3_CSV="oof/seed7_s42_s123_test_tta4.csv"
$PY tools/seed_average.py \
  --members cnxb_512_e16_ra_all_test_tta4.csv oof/cnxb_512_e16_ra_all_s42_test_tta4.csv "$S123_CSV" \
  --weights 1 1 1 --out "$AVG3_CSV" \
  || { echo "3-seed 平均 FAILED rc=$?"; exit 1; }

echo "=== [5] 3-seed soup（1:1:1 权重平均）→ test TTA4 ==="
SOUP3_CSV="oof/soup3_test_tta4.csv"
if [ -f "$SOUP3" ]; then
  echo "已存在 $SOUP3，跳过 soup 权重平均"
else
  mkdir -p "$(dirname "$SOUP3")"
  $PY tools/seed_soup.py --ckpts "$S7" "$S42" "$OUT/last.pt" --weights 1 1 1 --out "$SOUP3" \
    || { echo "seed_soup FAILED rc=$?"; exit 1; }
fi
$PY tools/oof_predict.py --ckpt "$SOUP3" \
  --csv glaucoma_first/test.csv --cache-root "$CACHE" --split-dir test \
  --img-size 512 --tta 4 --use ema --out "$SOUP3_CSV" \
  || { echo "soup3 PREDICT FAILED rc=$?"; exit 1; }

echo "=== [6] 与 ens10 融合（w=$WEIGHTS，logit + rank）==="
for C in "$S123_CSV" "$AVG3_CSV" "$SOUP3_CSV"; do
  [ -f "$C" ] || { echo "ERROR: 缺少 $C"; exit 1; }
  $PY tools/fusion_ens10.py --ens10 "$ENS10" --new "$C" --out-dir oof --weights $WEIGHTS \
    || echo "WARN: 融合失败 $C"
done

echo "=== [7] 强制校验（5855/顺序/NaN/mean/pos + Spearman/top200）==="
$PY - "$S123_CSV" "$AVG3_CSV" "$SOUP3_CSV" <<'PYEOF'
import csv, glob, math, os, sys

def rd(p):
    with open(p, encoding='utf-8', newline='') as f:
        return [(x['Image_name'], float(x['Probability'])) for x in csv.DictReader(f)]

def rank(v):
    n = len(v); o = sorted(range(n), key=lambda i: v[i]); r = [0.0]*n; i = 0
    while i < n:
        j = i
        while j+1 < n and v[o[j+1]] == v[o[i]]:
            j += 1
        a = (i+j)/2.0 + 1.0
        for k in range(i, j+1):
            r[o[k]] = a
        i = j+1
    return r

def pear(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / math.sqrt(
        sum((a-mx)**2 for a in x) * sum((b-my)**2 for b in y))

def sp(a, b):
    return pear(rank(a), rank(b))

def top200(names, probs):
    return set(names[i] for i in sorted(range(len(probs)), key=lambda i: -probs[i])[:200])

official = [x['Image_name'] for x in csv.DictReader(open('glaucoma_first/test.csv', encoding='utf-8'))]
REF_CANDS = [p for p in ('seed7_s42_ens10_logit_w20.csv', 'cnxb_512_e16_ra_all_ens10_logit_w10.csv') if os.path.exists(p)]
MEMBERS = {'seed7': 'cnxb_512_e16_ra_all_test_tta4.csv',
           'seed42': 'cnxb_512_e16_ra_all_s42_test_tta4.csv'}
MEMBERS['s123'] = sys.argv[1]

refs = {os.path.basename(p): rd(p) for p in REF_CANDS}
mem = {k: rd(v) for k, v in MEMBERS.items() if os.path.exists(v)}

cands = []
for p in [sys.argv[1], sys.argv[2], sys.argv[3]]:
    cands += [p]
cands += sorted(glob.glob('oof/*s123*_ens10_*.csv')) + sorted(glob.glob('oof/soup3_test_tta4_ens10_*.csv'))

print('\n--- 基础校验 ---')
ok_all = True
for p in dict.fromkeys(cands):
    rows = rd(p); names = [n for n, _ in rows]; probs = [v for _, v in rows]
    bad = sum(1 for v in probs if v != v or v < 0 or v > 1)
    ok = (len(rows) == 5855) and (names == official) and bad == 0
    ok_all = ok_all and ok
    print('%-58s rows=%d order=%s bad=%d mean=%.5f pos=%d %s'
          % (os.path.basename(p), len(rows), names == official, bad,
             sum(probs)/len(probs), sum(1 for v in probs if v >= 0.5),
             'OK' if ok else 'FAIL'))

print('\n--- 成员一致性（vs seed7 / seed42，按名字对齐）---')
for label, p in (('s123', sys.argv[1]), ('3seed_avg', sys.argv[2]), ('soup3', sys.argv[3])):
    if not os.path.exists(p):
        continue
    rows = dict(rd(p)); names = [n for n, _ in rd(p)]
    v = [rows[n] for n in names]
    t = top200(names, v)
    for k, rf in mem.items():
        m = dict(rf); rv = [m[n] for n in names]
        tr = top200(names, rv)
        print('  %-10s vs %-6s Spearman=%.5f  top200=%d  mean(new)=%.5f mean(ref)=%.5f'
              % (label, k, sp(v, rv), len(t & tr), sum(v)/len(v), sum(rv)/len(rv)))

print('\n--- 每个候选 vs 线上 98.84 参照 ---')
for rfname, rf in refs.items():
    rm = dict(rf); rnames = [n for n, _ in rf]; rp = [rm[n] for n in rnames]
    rt = top200(rnames, rp)
    print('  参照=%s (mean=%.5f)' % (rfname, sum(rp)/len(rp)))
    for p in dict.fromkeys(cands):
        rows = rd(p); names = [n for n, _ in rows]; m = dict(rows)
        v = [m[n] for n in rnames]
        print('    %-56s rho=%6.4f top200=%3d mean=%.5f'
              % (os.path.basename(p), sp(rp, v), len(rt & top200(rnames, v)), sum(v)/len(v)))
print('\nNOTE: 与 98.84 参照 rho 越接近 = 改动越小；提分需要「有效新信息」而非完全重合。')
if not ok_all:
    print('RESULT: FAIL（有候选未通过 5855/顺序/NaN 校验）'); sys.exit(1)
print('RESULT: PASS（全部候选通过基础校验）')
PYEOF
RC=$?
[ "$RC" -eq 0 ] || { echo "ERROR: 校验未通过"; exit 1; }

echo "=== 交付物清单 ==="
echo "  $OUT/{last,best}.pt（权重，上传 Release，勿入库）"
echo "  $S123_CSV"
echo "  $AVG3_CSV"
echo "  $SOUP3 与 $SOUP3_CSV"
ls -1 oof/*s123*_ens10_*.csv oof/soup3_test_tta4_ens10_*.csv 2>/dev/null
echo "  -> 打包推荐候选：python package_submission.py --csv <候选> --name submission_20261027_ra_all_s123_w10 --out-root <盘上目录>"
echo "  -> 回填 docs/ISSUE_152_RESULTS.md + 本 issue（md5/校验数字/推荐理由）；不要自动提交"
echo "===== #152 pipeline DONE $(date) ====="
