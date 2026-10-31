"""#193 headless runner: ConvNeXt-Large@384 + RA + train-on-all, seed 7.

Runs in a Kaggle script kernel (GPU + internet). One session = ~8h of training,
then saves resume.pt so the next session can continue with --resume-from.
"""
import glob
import os
import signal
import subprocess
import sys
import time

WORK = "/kaggle/working"
REPO = os.path.join(WORK, "glaucoma_first")
BRANCH = "main"
RUN = os.path.join(WORK, "runs", "cnxl_384_e16_ra_all_s7")
TRAIN_BUDGET = 29400  # 8h10m; session 2 has ~7.3h of training left, leaves room for infer + output


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


def find_cache():
    cands = [
        "/kaggle/input/datasets/linglingyusc/glaucoma/cache_all",
        "/kaggle/input/glaucoma/cache_all",
    ]
    cands += sorted(glob.glob("/kaggle/input/*/cache_all"))
    cands += sorted(glob.glob("/kaggle/input/*/*/cache_all"))
    for c in cands:
        if os.path.isdir(os.path.join(c, "train")) and os.path.isdir(os.path.join(c, "test")):
            return c
    raise SystemExit("cache_all not found. /kaggle/input = %s" % glob.glob("/kaggle/input/*"))


def find_resume():
    """resume.pt 可能在上一场发布的产物里（作为数据集挂到 /kaggle/input）。"""
    local = os.path.join(RUN, "resume.pt")
    if os.path.isfile(local):
        return local
    # 两套挂载布局都在用：/kaggle/input/<slug>/ 和 /kaggle/input/datasets/<owner>/<slug>/
    for root, dirs, files in os.walk("/kaggle/input"):
        if root.count(os.sep) - "/kaggle/input".count(os.sep) > 3:
            dirs[:] = []
            continue
        if "resume.pt" in files:
            return os.path.join(root, "resume.pt")
    return None


def main():
    print("cwd:", os.getcwd(), flush=True)
    print("nvidia-smi:", flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])

    cache = find_cache()
    n_tr = len(os.listdir(os.path.join(cache, "train")))
    n_te = len(os.listdir(os.path.join(cache, "test")))
    print(f"cache={cache} train={n_tr} test={n_te}", flush=True)
    if n_tr != 57245 or n_te != 5855:
        print(f"!! unexpected image counts (want 57245/5855)", flush=True)

    if not os.path.isdir(REPO):
        sh(["git", "clone", "-q", "https://github.com/dongmingkun/glaucoma_first.git", REPO], check=True)
    sh(["git", "-C", REPO, "fetch", "-q", "origin", BRANCH], check=True)
    sh(["git", "-C", REPO, "checkout", "-q", BRANCH], check=True)
    sh(["git", "-C", REPO, "reset", "-q", "--hard", "origin/" + BRANCH], check=True)
    head = subprocess.check_output(["git", "-C", REPO, "log", "--oneline", "-1"]).decode().strip()
    print("HEAD:", head, flush=True)

    sh([sys.executable, "-m", "pip", "install", "-q", "timm", "scikit-learn", "pandas", "pillow"], check=True)

    os.makedirs(RUN, exist_ok=True)
    resume = find_resume()
    extra = ["--resume-from", resume] if resume else []
    print("resume:", resume if resume else "none (fresh start)", flush=True)

    cmd = [
        sys.executable, "train.py",
        "--arch", "convnext_large", "--img-size", "384",
        "--cache-root", cache,
        "--train-on-all", "--epochs", "16", "--batch-size", "8", "--accum", "6",
        "--lr", "5e-5", "--head-lr", "2.5e-4", "--pretrained", "1",
        "--randaugment", "2", "9", "--seed", "7",
        "--loader", "workers", "--workers", "2",
        "--save-every-steps", "0",
        "--out-dir", RUN, "--tag", "cnxl_384_e16_ra_all_s7",
    ] + extra

    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=REPO)
    try:
        rc = p.wait(timeout=TRAIN_BUDGET)
    except subprocess.TimeoutExpired:
        print(f"[runner] training budget hit, SIGTERM -> train.py saves resume.pt", flush=True)
        p.send_signal(signal.SIGTERM)
        try:
            rc = p.wait(timeout=900)
        except subprocess.TimeoutExpired:
            print("[runner] SIGTERM ignored, killing", flush=True)
            p.kill()
            rc = -9
    print(f"[runner] train exit={rc} elapsed={(time.time() - t0) / 3600:.2f}h", flush=True)

    log = os.path.join(RUN, "log.csv")
    if os.path.isfile(log):
        with open(log, "r", encoding="utf-8", errors="replace") as f:
            print("--- train log ---\n" + f.read(), flush=True)

    ck = os.path.join(RUN, "best.pt")
    if os.path.isfile(ck):
        out = os.path.join(WORK, "cnxl_384_e16_ra_all_s7_test_tta4.csv")
        sh([sys.executable, "infer.py", "--ckpt", ck, "--split", "test", "--tta", "4",
            "--cache-root", cache, "--workers", "2", "--batch-size", "16",
            "--out", out], cwd=REPO, check=False)
    else:
        print("[runner] no best.pt, skip infer", flush=True)

    print("--- /kaggle/working ---", flush=True)
    for root, _, files in os.walk(WORK):
        for fn in files:
            p_ = os.path.join(root, fn)
            print(f"{os.path.getsize(p_) / 2**30:7.2f}G  {p_}", flush=True)


if __name__ == "__main__":
    main()
