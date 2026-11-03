"""#199 headless runner（阶段 1）：ConvNeXtV2-Base@512 + RA + train-only，seed 42。

Kaggle script kernel（GPU + internet）跑 #199 阶段 1：
16ep 训练（干净 valid，每 epoch validation-tta4）→ infer.py valid TTA4 →
门槛一自检（valid AUROC >= 0.9920）→ gate_199_valid.py（复核+Spearman；proxy 缺席如实标 MISSING_PROXY）。

数据探测顺序（关键：阶段 1 必须用 train/valid 分离的 cache512，57245 的 cache_all 会泄漏 valid）：
  1) 任意挂载数据集下的 cache512（train=51396 且 valid=5849）
  2) 原始图 glaucoma_first/{train,valid}（在 Kaggle 上现 build cache512，约 1-2h）
预算：默认 10.8h 训练预算，超时 SIGTERM -> train.py 落 resume.pt（#193 已验证的续传协议）。
"""
import glob
import json
import os
import signal
import subprocess
import sys
import time
import traceback

WORK = "/kaggle/working"
REPO = "/tmp/glaucoma_first"
BRANCH = "main"
RUN = os.path.join(WORK, "runs", "cnxv2b_512_e16_ra")
TAG = "cnxv2b_512_e16_ra"
TRAIN_BUDGET = int(float(os.environ.get("TRAIN_BUDGET_H", "10.8")) * 3600)
GATE1 = 0.9920


def _tee_all_output():
    """所有 runner/subprocess 输出同步写入 WORK/runner.log（崩溃日志随 kernel output 带回）。"""
    logf = open(os.path.join(WORK, "runner.log"), "a", buffering=1)
    os.dup2(logf.fileno(), 1)
    os.dup2(logf.fileno(), 2)
    print(f"[runner] logging tee -> {logf.name}", flush=True)


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


def reconstruct_from_cache_all():
    """cache_all（train/valid 同为 57245 混合目录）按文件名前缀精确解出 cache512：
    官方 train 图全部为 train_*.jpg（51396）、valid 图全部为 valid_*.jpg（5849）
    （cache_all 由 cp -rs cache512/{train,valid} 合并而成，文件名保留原前缀）。
    重建后逐项计数硬校验，任何一项不符即放弃。"""
    cands = ["/kaggle/input/glaucoma/cache_all"]
    cands += sorted(glob.glob("/kaggle/input/*/cache_all"))
    cands += sorted(glob.glob("/kaggle/input/*/*/cache_all"))
    src = None
    for c in cands:
        if (os.path.isdir(os.path.join(c, "train")) and os.path.isdir(os.path.join(c, "test"))
                and len(os.listdir(os.path.join(c, "train"))) == 57245):
            src = c
            break
    if not src:
        return None
    print(f"[cache] cache_all @ {src}（57245 混合）-> 按前缀解出 cache512", flush=True)
    out = os.path.join(WORK, "cache512")
    for sub in ("train", "valid", "test"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    stats = {"train": 0, "valid": 0}
    t_dir = os.path.join(src, "train")
    for fn in os.listdir(t_dir):
        stem, ext = os.path.splitext(fn)
        sub = "train" if stem.startswith("train_") else ("valid" if stem.startswith("valid_") else None)
        if sub is None:
            print(f"[cache] WARN: 未识别前缀 {fn}，放弃重建", flush=True)
            return None
        link = os.path.join(out, sub, fn)
        if not os.path.exists(link):
            try:
                os.symlink(os.path.join(t_dir, fn), link)
            except OSError:
                import shutil
                shutil.copy2(os.path.join(t_dir, fn), link)
        stats[sub] += 1
    for fn in os.listdir(os.path.join(src, "test")):
        link = os.path.join(out, "test", fn)
        if not os.path.exists(link):
            try:
                os.symlink(os.path.join(src, "test", fn), link)
            except OSError:
                import shutil
                shutil.copy2(os.path.join(src, "test", fn), link)
    n_tr = len(os.listdir(os.path.join(out, "train")))
    n_va = len(os.listdir(os.path.join(out, "valid")))
    n_te = len(os.listdir(os.path.join(out, "test")))
    print(f"[cache] 重建完成：train={n_tr} valid={n_va} test={n_te}（期望 51396/5849/5855，分拣 {stats}）",
          flush=True)
    if n_tr == 51396 and n_va == 5849 and n_te == 5855:
        return out
    print("[cache] ERROR: 重建计数不符，放弃", flush=True)
    return None


def find_cache512():
    """阶段 1 必须 train/valid 分离（51396/5849）——cache_all 的 57245 混合目录会造成 valid 泄漏。"""
    cands = ["/kaggle/input/glaucoma/cache512"]
    cands += sorted(glob.glob("/kaggle/input/*/cache512"))
    cands += sorted(glob.glob("/kaggle/input/*/*/cache512"))
    cands += sorted(glob.glob("/kaggle/input/*/datasets/*/cache512"))
    for c in cands:
        if (os.path.isdir(os.path.join(c, "train")) and os.path.isdir(os.path.join(c, "valid"))):
            n_tr = len(os.listdir(os.path.join(c, "train")))
            n_va = len(os.listdir(os.path.join(c, "valid")))
            if n_tr == 51396 and n_va == 5849:
                print(f"[cache] {c} train={n_tr} valid={n_va} OK", flush=True)
                return c
            print(f"[cache] {c} 计数 {n_tr}/{n_va} 不符，跳过（防泄漏）", flush=True)
    return None


def build_cache512():
    """原始图兜底：用仓库 preprocess.py 现建 cache512（--data-root 指向挂载的原始图）。"""
    raw = None
    for pat in ("/kaggle/input/*/glaucoma_first", "/kaggle/input/glaucoma_first"):
        for d in glob.glob(pat):
            if os.path.isdir(os.path.join(d, "train")) and os.path.isdir(os.path.join(d, "valid")):
                raw = d
                break
    if not raw:
        return None
    print(f"[cache] 原始图 {raw} -> 现建 cache512（约 1-2h）", flush=True)
    out = os.path.join(WORK, "cache512")
    r = sh([sys.executable, "preprocess.py", "--data-root", raw, "--out", out,
            "--size", "512", "--workers", "4"], cwd=REPO)
    if r.returncode != 0 or not os.path.isdir(os.path.join(out, "train")):
        return None
    return out if len(os.listdir(os.path.join(out, "train"))) == 51396 else None


def find_resume():
    local = os.path.join(RUN, "resume.pt")
    if os.path.isfile(local):
        return local
    for root, dirs, files in os.walk("/kaggle/input"):
        if root.count(os.sep) - "/kaggle/input".count(os.sep) > 3:
            dirs[:] = []
            continue
        if "resume.pt" in files:
            return os.path.join(root, "resume.pt")
    return None


def main():
    print("cwd:", os.getcwd(), flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])

    if not os.path.isdir(REPO):
        sh(["git", "clone", "-q", "https://github.com/dongmingkun/glaucoma_first.git", REPO], check=True)
    sh(["git", "-C", REPO, "fetch", "-q", "origin", BRANCH], check=True)
    sh(["git", "-C", REPO, "checkout", "-q", BRANCH], check=True)
    sh(["git", "-C", REPO, "reset", "-q", "--hard", "origin/" + BRANCH], check=True)
    head = subprocess.check_output(["git", "-C", REPO, "log", "--oneline", "-1"]).decode().strip()
    print("HEAD:", head, flush=True)

    sh([sys.executable, "-m", "pip", "install", "-q", "timm", "scikit-learn"], check=True)

    cache = find_cache512() or reconstruct_from_cache_all() or build_cache512()
    if not cache:
        raise SystemExit("未找到/未能构建 train=51396+valid=5849 的 cache512；"
                         "/kaggle/input = %s" % glob.glob("/kaggle/input/*")[:10])
    os.makedirs(RUN, exist_ok=True)

    resume = find_resume()
    extra = ["--resume-from", resume] if resume else []
    print("resume:", resume if resume else "none (fresh start)", flush=True)

    cmd = [
        sys.executable, "train.py",
        "--arch", "convnextv2_base", "--img-size", "512",
        "--cache-root", cache,
        "--epochs", "16", "--batch-size", "8", "--accum", "6",
        "--lr", "1e-4", "--head-lr", "5e-4", "--randaugment", "2", "9", "--pretrained", "1",
        "--seed", "42", "--loader", "workers", "--workers", "4",
        "--validation-tta", "4", "--save-every-steps", "200",
        "--tag", TAG, "--out-dir", RUN,
    ] + extra
    print("TRAIN CMD:", " ".join(cmd), flush=True)

    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=REPO)
    try:
        rc = p.wait(timeout=TRAIN_BUDGET)
    except subprocess.TimeoutExpired:
        print("[runner] budget hit, SIGTERM -> train.py saves resume.pt", flush=True)
        p.send_signal(signal.SIGTERM)
        try:
            rc = p.wait(timeout=900)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = -9
    print(f"[runner] train exit={rc} elapsed={(time.time() - t0) / 3600:.2f}h", flush=True)

    log = os.path.join(RUN, "log.csv")
    if os.path.isfile(log):
        content = open(log, encoding="utf-8", errors="replace").read()
        print("--- train log ---\n" + content, flush=True)
        with open(os.path.join(WORK, "log.csv"), "w", encoding="utf-8") as f:
            f.write(content)

    last = os.path.join(RUN, "last.pt")
    valid_csv = os.path.join(WORK, "cnxv2b_512_e16_ra_valid_tta4.csv")
    auroc = None
    if os.path.isfile(last):
        r = sh([sys.executable, "infer.py", "--ckpt", last, "--split", "valid", "--tta", "4",
                "--use", "ema", "--cache-root", cache, "--workers", "2", "--batch-size", "16",
                "--out", valid_csv], cwd=REPO, check=False)
        if r.returncode == 0 and os.path.isfile(valid_csv):
            import csv as _csv
            import numpy as np
            from sklearn.metrics import roc_auc_score
            lab = os.path.join(REPO, "glaucoma_first", "valid.csv")
            names = [row["Image_name"] for row in _csv.DictReader(open(lab, encoding="utf-8-sig"))]
            y = np.array([int(row["True_Label"]) for row in _csv.DictReader(open(lab, encoding="utf-8-sig"))])
            d = {row["Image_name"]: float(row["Probability"])
                 for row in _csv.DictReader(open(valid_csv, encoding="utf-8-sig"))}
            pr = np.array([d[n] for n in names])
            auroc = float(roc_auc_score(y, pr))
            g1 = "PASS" if auroc >= GATE1 else "FAIL"
            print(f"[gate1] valid TTA4(EMA) AUROC={auroc:.5f}  门槛>={GATE1}  ->  {g1}", flush=True)
            json.dump({"gate1_auroc": auroc, "threshold": GATE1, "pass": auroc >= GATE1},
                      open(os.path.join(WORK, "gate1.json"), "w"), indent=1)
            # 门槛脚本：复核 + Spearman（--proxy 缺席时如实标 MISSING_PROXY，供队长补）
            for ref in glob.glob("/kaggle/input/*/*valid*tta4*.csv")[:4]:
                print("[ref-candidate]", ref, flush=True)
            sh([sys.executable, "tools/gate_199_valid.py", "--csv", valid_csv,
                "--json", os.path.join(WORK, "gate_199.json")], cwd=REPO, check=False)
    else:
        print("[runner] no last.pt（训练中断且无 epoch 末落盘），本会话无 valid 产物", flush=True)

    print("--- /kaggle/working ---", flush=True)
    for root, _, files in os.walk(WORK):
        for fn in files:
            p_ = os.path.join(root, fn)
            print(f"{os.path.getsize(p_) / 2**30:7.2f}G  {p_}", flush=True)
    print("[runner] DONE", flush=True)


if __name__ == "__main__":
    _tee_all_output()
    try:
        main()
    except BaseException:
        traceback.print_exc()
        print("[runner] FATAL — traceback 已写入上方与本文件", flush=True)
        raise
