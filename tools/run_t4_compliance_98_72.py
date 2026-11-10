"""#211 98.72 三模型合规包 T4 实测：下载/校验、100 图 F1/F2、全量 5855 出 CSV。

不读 test 标签；test.csv 只用 Image_name/Age 出概率。
"""
import csv
import glob
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request

WORK = "/kaggle/working"
PKG = os.path.join(WORK, "pkg")
COMMIT = "96b2d71"
RAW = "https://raw.githubusercontent.com/dongmingkun/glaucoma_first/%s/" % COMMIT
ASSETS = [
    ("weights-1016", "convnext_base_384_cache512_1015_auprc0.8644.pt", "cb384.pt",
     176_328_125, "a389f41cb367132b4c23e2f9195bf15ac87c9df1214d359656f36ee63aff710e"),
    ("weights-1016", "cs_hi512_all_fp16.pt", "cs_hi512_all.pt",
     199_660_651, "3e04d1906965742d8b13657b16e539063b05dab622c5272454939f57105f492e"),
    ("compliant_pkg-1028", "cs_hi512_all_ra.pt", "cs_hi512_all_ra.pt",
     99_831_694, "5a04ebe3152416022b0e07c9d42f7af57532b9cb366771c09889173a154c86e6"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, dst):
    tmp = dst + ".part"
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        n = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            n += len(b)
            if n % (50 << 20) < (1 << 20):
                print("  %.0f MB" % (n / 1e6), flush=True)
    os.replace(tmp, dst)
    return os.path.getsize(dst)


def main():
    os.makedirs(PKG, exist_ok=True)
    print("torch:", __import__("torch").__version__, "cuda:", __import__("torch").cuda.get_device_name(0), flush=True)

    fetch(RAW + "submit_pkg/predict.py", os.path.join(PKG, "predict.py"))
    fetch(RAW + "glaucoma_first/test.csv", os.path.join(PKG, "test.csv"))
    for tag, asset, local, size, want in ASSETS:
        dst = os.path.join(PKG, local)
        if not (os.path.isfile(dst) and os.path.getsize(dst) == size and sha256(dst) == want):
            print("download", asset, flush=True)
            got_n = fetch("https://github.com/dongmingkun/glaucoma_first/releases/download/%s/%s" % (tag, asset), dst)
            assert got_n == size, (local, got_n, size)
        got = sha256(dst)
        assert got == want, (local, got, want)
        print("OK", local, size, got, flush=True)

    cands = ["/kaggle/input/datasets/linglingyusc/glaucoma/cache_all/test",
             "/kaggle/input/glaucoma/cache_all/test"]
    cands += sorted(glob.glob("/kaggle/input/*/cache_all/test"))
    cands += sorted(glob.glob("/kaggle/input/*/*/cache_all/test"))
    img_dir = None
    for c in cands:
        if os.path.isdir(c) and len(glob.glob(os.path.join(c, "*.jpg"))) == 5855:
            img_dir = c
            break
    if img_dir is None:
        raise SystemExit("test cache not found: %s" % cands)
    print("images:", img_dir, flush=True)

    rows = list(csv.DictReader(open(os.path.join(PKG, "test.csv"), encoding="utf-8-sig", newline="")))
    assert len(rows) == 5855, len(rows)

    # 100-image subset (same order as test.csv)
    sub_dir = os.path.join(WORK, "images100")
    os.makedirs(sub_dir, exist_ok=True)
    for name in os.listdir(sub_dir):
        os.remove(os.path.join(sub_dir, name))
    sub100 = rows[:100]
    for r in sub100:
        src = os.path.join(img_dir, r["Image_name"])
        assert os.path.isfile(src), src
        os.symlink(src, os.path.join(sub_dir, r["Image_name"]))
    with open(os.path.join(WORK, "test100.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image_name", "Age"])
        for r in sub100:
            w.writerow([r["Image_name"], r["Age"]])

    weights = [os.path.join(PKG, a[2]) for a in ASSETS]
    base = [sys.executable, os.path.join(PKG, "predict.py"), "--csv"]
    common = ["--weights"] + weights + ["--tta", "1", "--batch-size", "1", "--prep-workers", "8"]
    metrics = {"device": __import__("torch").cuda.get_device_name(0),
               "torch": __import__("torch").__version__,
               "assets": [{"name": a[2], "bytes": a[3], "sha256": a[4]} for a in ASSETS]}

    # warmup (5 images) then 3 timed 100-image runs
    sub5 = os.path.join(WORK, "test5.csv")
    with open(sub5, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image_name", "Age"])
        for r in sub100[:5]:
            w.writerow([r["Image_name"], r["Age"]])
    t = time.time()
    subprocess.run(base + [sub5, "--images", sub_dir, "--out", os.path.join(WORK, "warm5.csv")] + common,
                   check=True)
    print("warmup s=", round(time.time() - t, 2), flush=True)

    runs = []
    for k in range(3):
        out = os.path.join(WORK, "pred100_run%d.csv" % (k + 1))
        t = time.time()
        subprocess.run(base + [os.path.join(WORK, "test100.csv"), "--images", sub_dir, "--out", out] + common,
                       check=True)
        dt = time.time() - t
        runs.append(dt)
        print("run%d s=%.2f  %.1f ms/img" % (k + 1, dt, dt / 100 * 1000), flush=True)
    metrics["f1_100"] = {"seconds": [round(x, 3) for x in runs],
                         "mean_ms_per_img": round(statistics.mean(runs) / 100 * 1000, 2),
                         "min_ms_per_img": round(min(runs) / 100 * 1000, 2)}

    # cost-like path: import predict.py, per-image batch=1, prep_workers=1
    spec = importlib.util.spec_from_file_location("pkgpredict", os.path.join(PKG, "predict.py"))
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    groups = {}
    for p, wgt in zip(weights, [1 / 3] * 3):
        m, meta = pp.load_model(p, "cuda")
        groups.setdefault(meta["cache_size"], []).append((m, meta, wgt))
        print("model", os.path.basename(p), meta, flush=True)
    pp.run_group([os.path.join(sub_dir, sub100[0]["Image_name"])], [44.0],
                 list(groups.values())[0], "cuda", 1, 1, 1)  # warm kernels

    def predict_one(path, age):
        ens = 0.0
        for cs, members in groups.items():
            probs = pp.run_group([path], [age], members, "cuda", 1, 1, 1)
            for (m, meta, wgt), pr in zip(members, probs):
                ens += wgt * float(pr[0])
        return ens

    times, cost_probs = [], []
    for r in sub100:
        t = time.perf_counter()
        cost_probs.append(predict_one(os.path.join(sub_dir, r["Image_name"]), pp.clean_age(r.get("Age"))))
        times.append((time.perf_counter() - t) * 1000)
    times_sorted = sorted(times)
    metrics["f1_cost_100"] = {"p50_ms": round(statistics.median(times), 2),
                              "p95_ms": round(times_sorted[int(0.95 * (len(times) - 1))], 2),
                              "mean_ms": round(statistics.mean(times), 2)}
    print("cost path:", metrics["f1_cost_100"], flush=True)

    with open(os.path.join(WORK, "cost100.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image_name", "Probability", "Pred_Label"])
        for r, p in zip(sub100, cost_probs):
            w.writerow([r["Image_name"], "%.6f" % p, 1 if p >= 0.5 else 0])

    pred = {}
    for r in csv.DictReader(open(os.path.join(WORK, "pred100_run1.csv"), encoding="utf-8-sig", newline="")):
        pred[r["Image_name"]] = float(r["Probability"])
    diffs = [abs(pred[r["Image_name"]] - p) for r, p in zip(sub100, cost_probs)]
    bin_mis = sum(1 for r, p in zip(sub100, cost_probs) if (pred[r["Image_name"]] >= 0.5) != (p >= 0.5))
    metrics["f2_100"] = {"binary_mismatch": bin_mis, "max_abs_diff": max(diffs),
                         "mean_abs_diff": statistics.mean(diffs)}
    print("F2:", metrics["f2_100"], flush=True)

    # full 5855 run for md5 + end-to-end image count
    full = os.path.join(PKG, "test.csv")
    t = time.time()
    out_full = os.path.join(WORK, "submission.csv")
    subprocess.run(base + [full, "--images", img_dir, "--out", out_full] + common, check=True)
    dt = time.time() - t
    data = open(out_full, "rb").read()
    n_rows = sum(1 for _ in csv.DictReader(open(out_full, encoding="utf-8-sig", newline="")))
    metrics["full"] = {"rows": n_rows, "seconds": round(dt, 1),
                       "ms_per_img": round(dt / n_rows * 1000, 2),
                       "md5_crlf": hashlib.md5(data).hexdigest(),
                       "sha256_crlf": hashlib.sha256(data).hexdigest(),
                       "bytes": len(data)}
    print("full:", metrics["full"], flush=True)

    metrics["package"] = {"files": [{"name": a[2], "bytes": a[3], "sha256": a[4]} for a in ASSETS] +
                                   [{"name": "predict.py", "bytes": os.path.getsize(os.path.join(PKG, "predict.py")),
                                     "sha256": sha256(os.path.join(PKG, "predict.py"))}],
                          "total_bytes": sum(a[3] for a in ASSETS) + os.path.getsize(os.path.join(PKG, "predict.py"))}
    json.dump(metrics, open(os.path.join(WORK, "metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
