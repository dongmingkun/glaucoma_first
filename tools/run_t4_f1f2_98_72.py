"""#213 按组委会 F1/F2 口径复测 98.72 三模型包（T4）。

F1：batch=1 串行，100 张连续 3 轮，逐图 3 轮取均值再对 100 张取平均；
    含预处理与后处理，不含模型加载与预热；报告 mean/p50/p95。
F2：cost 路径（导入 predict.py 同一 load_model/run_group） vs predict.py CLI，
    100 张逐张比对：二值不一致、概率排序不一致、max|Δp|。
只读 test.csv 的 Image_name/Age，不读标签。
"""
import csv
import glob
import hashlib
import importlib.util
import json
import os
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
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(tmp, dst)
    return os.path.getsize(dst)


def main():
    os.makedirs(PKG, exist_ok=True)
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.get_device_name(0), flush=True)

    fetch(RAW + "submit_pkg/predict.py", os.path.join(PKG, "predict.py"))
    fetch(RAW + "glaucoma_first/test.csv", os.path.join(PKG, "test.csv"))
    for tag, asset, local, size, want in ASSETS:
        dst = os.path.join(PKG, local)
        if not (os.path.isfile(dst) and os.path.getsize(dst) == size and sha256(dst) == want):
            fetch("https://github.com/dongmingkun/glaucoma_first/releases/download/%s/%s" % (tag, asset), dst)
        assert os.path.getsize(dst) == size and sha256(dst) == want, local
        print("OK", local, flush=True)

    img_dir = None
    for c in ["/kaggle/input/datasets/linglingyusc/glaucoma/cache_all/test",
              "/kaggle/input/glaucoma/cache_all/test"] + sorted(glob.glob("/kaggle/input/*/cache_all/test")):
        if os.path.isdir(c) and len(glob.glob(os.path.join(c, "*.jpg"))) == 5855:
            img_dir = c
            break
    assert img_dir, "test cache not found"
    print("images:", img_dir, flush=True)

    rows = list(csv.DictReader(open(os.path.join(PKG, "test.csv"), encoding="utf-8-sig", newline="")))
    assert len(rows) == 5855
    sub100 = rows[:100]
    sub_dir = os.path.join(WORK, "images100")
    os.makedirs(sub_dir, exist_ok=True)
    for r in sub100:
        dst = os.path.join(sub_dir, r["Image_name"])
        if not os.path.exists(dst):
            os.symlink(os.path.join(img_dir, r["Image_name"]), dst)
    test100 = os.path.join(WORK, "test100.csv")
    with open(test100, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Image_name", "Age"])
        for r in sub100:
            w.writerow([r["Image_name"], r["Age"]])

    spec = importlib.util.spec_from_file_location("pkgpredict", os.path.join(PKG, "predict.py"))
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    weights = [os.path.join(PKG, a[2]) for a in ASSETS]
    groups = {}
    for p, wgt in zip(weights, [1 / 3] * 3):
        m, meta = pp.load_model(p, "cuda")
        groups.setdefault(meta["cache_size"], []).append((m, meta, wgt))
        print("model", os.path.basename(p), meta, flush=True)
    members = list(groups.values())[0]

    def predict_one(path, age):
        probs = pp.run_group([path], [age], members, "cuda", 1, 1, 1)
        return sum(wgt * float(pr[0]) for (m, meta, wgt), pr in zip(members, probs))

    # warmup (excluded), then 3 rounds per-image timing
    for r in sub100[:5]:
        predict_one(os.path.join(sub_dir, r["Image_name"]), pp.clean_age(r.get("Age")))
    torch.cuda.synchronize()

    per_round = []
    for k in range(3):
        times, probs = [], []
        for r in sub100:
            p = os.path.join(sub_dir, r["Image_name"])
            age = pp.clean_age(r.get("Age"))
            torch.cuda.synchronize()
            t = time.perf_counter()
            v = predict_one(p, age)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000)
            probs.append(v)
        per_round.append(times)
        print("round%d mean=%.2f p50=%.2f p95=%.2f" % (
            k + 1, statistics.mean(times), statistics.median(times),
            sorted(times)[int(0.95 * (len(times) - 1))]), flush=True)
        if k == 0:
            with open(os.path.join(WORK, "cost100.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Image_name", "Probability", "Pred_Label"])
                for r, v in zip(sub100, probs):
                    w.writerow([r["Image_name"], "%.6f" % v, 1 if v >= 0.5 else 0])

    per_img = [statistics.mean(per_round[k][i] for k in range(3)) for i in range(100)]
    metrics = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "protocol": "batch=1 serial, 100 imgs x 3 rounds, per-image mean, excl. load/warmup",
        "f1_round_mean_ms": [round(statistics.mean(x), 2) for x in per_round],
        "f1_mean_ms": round(statistics.mean(per_img), 2),
        "f1_p50_ms": round(statistics.median(per_img), 2),
        "f1_p95_ms": round(sorted(per_img)[int(0.95 * 99)], 2),
        "f1_max_ms": round(max(per_img), 2),
    }
    print("F1:", metrics, flush=True)

    # F2: predict.py CLI on same 100 images
    out_cli = os.path.join(WORK, "pred100.csv")
    subprocess.run([sys.executable, os.path.join(PKG, "predict.py"), "--csv", test100,
                    "--images", sub_dir, "--weights"] + weights +
                   ["--tta", "1", "--batch-size", "1", "--prep-workers", "1",
                    "--out", out_cli], check=True)
    cli = {}
    for r in csv.DictReader(open(out_cli, encoding="utf-8-sig", newline="")):
        cli[r["Image_name"]] = float(r["Probability"])
    cost = {}
    for r in csv.DictReader(open(os.path.join(WORK, "cost100.csv"), encoding="utf-8-sig", newline="")):
        cost[r["Image_name"]] = float(r["Probability"])
    names = [r["Image_name"] for r in sub100]
    cp = [cost[n] for n in names]
    pp_ = [cli[n] for n in names]
    diffs = [abs(a - b) for a, b in zip(cp, pp_)]
    bin_mis = sum(1 for a, b in zip(cp, pp_) if (a >= 0.5) != (b >= 0.5))
    inv = sum(1 for i in range(100) for j in range(i + 1, 100)
              if (cp[i] - cp[j]) * (pp_[i] - pp_[j]) < 0)
    order_a = sorted(range(100), key=lambda i: (cp[i], i))
    order_b = sorted(range(100), key=lambda i: (pp_[i], i))
    rank_a = {i: k for k, i in enumerate(order_a)}
    rank_b = {i: k for k, i in enumerate(order_b)}
    rank_mis = sum(1 for i in range(100) if rank_a[i] != rank_b[i])
    metrics["f2"] = {"binary_mismatch": bin_mis, "pairwise_inversions": inv,
                     "rank_position_mismatch": rank_mis,
                     "max_abs_diff": max(diffs), "mean_abs_diff": statistics.mean(diffs)}
    print("F2:", metrics["f2"], flush=True)
    json.dump(metrics, open(os.path.join(WORK, "metrics_f1f2.json"), "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
