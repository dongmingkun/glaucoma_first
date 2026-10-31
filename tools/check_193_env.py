"""#193 开跑前的环境预检（Kaggle / 本机通用，不依赖 torch 也能跑完大部分检查）。

用法：
  python tools/check_193_env.py --repo . --cache-root /kaggle/input/datasets/linglingyusc/glaucoma/cache_all

不给 --cache-root 时会自动在 /kaggle/input 下找 cache_all；找不到就报错退出。
全部通过才建议开 GPU 训练，否则先修环境——免费时长按小时扣，别拿训练去试环境。
"""
import argparse
import csv
import glob
import os
import shutil
import sys

EXPECT = {"train": 51396, "valid": 5849, "test": 5855}
TRAIN_ALL = EXPECT["train"] + EXPECT["valid"]   # 57245：train-on-all 的训练量


def _count_jpg(d):
    if not os.path.isdir(d):
        return -1
    with os.scandir(d) as it:
        return sum(1 for e in it if e.is_file() and e.name.lower().endswith(".jpg"))


def find_cache_root(explicit):
    cands = []
    if explicit:
        cands.append(explicit)
    cands += glob.glob("/kaggle/input/datasets/*/*/cache_all")
    cands += glob.glob("/kaggle/input/*/cache_all")
    cands += [os.environ.get("GLAUCOMA_CACHE_ROOT", "")]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "train")):
            return os.path.abspath(c)
    return None


def dir_size_gb(p):
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 2 ** 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="仓库根目录（含 glaucoma_first/train.csv）")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--work-dir", default="/kaggle/working")
    args = ap.parse_args()

    ok, warn = [], []
    print("=" * 68)

    # 1) CSV
    csv_root = os.path.join(args.repo, "glaucoma_first")
    for s in ("train", "valid", "test"):
        p = os.path.join(csv_root, s + ".csv")
        if not os.path.exists(p):
            ok.append(("csv", False, f"{s}.csv 缺失: {p}"))
            continue
        with open(p, encoding="utf-8-sig", newline="") as f:
            n = sum(1 for _ in csv.DictReader(f))
        hit = n == EXPECT[s]
        ok.append(("csv", hit, f"{s}.csv {n} 行 (期望 {EXPECT[s]})"))
    train_all = EXPECT["train"] + EXPECT["valid"]
    print(f"[csv] train+valid = {train_all} 行（--train-on-all 的实际训练量）")

    # 2) 缓存
    cache = find_cache_root(args.cache_root)
    if not cache:
        ok.append(("cache", False, "找不到 cache_all（给 --cache-root 或确认数据集已挂载）"))
        cache_for_cmd = "<CACHE_ROOT>"
    else:
        cache_for_cmd = cache
        n = {s: _count_jpg(os.path.join(cache, s)) for s in ("train", "valid", "test")}
        merged = n["train"] == TRAIN_ALL and n["valid"] in (-1, 0)
        split = n["train"] == EXPECT["train"] and n["valid"] == EXPECT["valid"]
        layout = "合并式(train=57245)" if merged else ("分离式(train/valid 分开)" if split else "不认识")
        ok.append(("cache", merged or split,
                   f"cache 布局={layout}: train={n['train']} valid={n['valid'] if n['valid'] >= 0 else '无'}"))
        ok.append(("cache", n["test"] == EXPECT["test"], f"cache/test {n['test']} 张 (期望 {EXPECT['test']})"))
        print(f"[cache] {cache}  ({dir_size_gb(cache):.2f} GB)")
        # MixedDS 找不到图时会回退到 test/（注释写的是伪标签/测试域），一旦真回退就是用测试图训练
        stems = []
        for s in ("train", "valid"):
            p = os.path.join(csv_root, s + ".csv")
            if os.path.exists(p):
                with open(p, encoding="utf-8-sig", newline="") as f:
                    stems += [r["Image_name"].rsplit(".", 1)[0] for r in csv.DictReader(f)]
        missing = [st for st in stems
                   if not (os.path.exists(os.path.join(cache, "train", st + ".jpg")) or
                           os.path.exists(os.path.join(cache, "valid", st + ".jpg")))]
        ok.append(("cache", not missing,
                   f"{len(stems)} 行训练样本中缺图 {len(missing)} 张"
                   + (f"，例: {missing[:3]}" if missing else "")))
        try:
            from PIL import Image
            sample = None
            for root, _, files in os.walk(os.path.join(cache, "train")):
                if files:
                    sample = os.path.join(root, files[0])
                    break
            if sample:
                im = Image.open(sample)
                print(f"[cache] 样本 {os.path.basename(sample)}: {im.size} {im.mode}")
                warn.append(("cache", im.size[0] < 384,
                             f"缓存短边 {im.size[0]} < 384，训练放大无新信息（检查是否为 512px 缓存）"))
        except ImportError:
            warn.append(("pil", False, "PIL 未安装，跳过样本尺寸检查"))

    # 3) 磁盘
    if os.path.isdir(args.work_dir):
        free = shutil.disk_usage(args.work_dir).free / 2 ** 30
        # resume.pt(模型+EMA+优化器) + last/best + CSV ≈ 2.5-3.5GB（Large）
        ok.append(("disk", free > 6, f"{args.work_dir} 剩余 {free:.1f} GB（resume.pt 约 1.4GB，留 6GB+）"))

    # 4) GPU / 依赖
    try:
        import torch
        print(f"[torch] {torch.__version__} cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            print(f"[gpu] {name}  sm_{cap[0]}{cap[1]}  {vram:.1f} GB  x{torch.cuda.device_count()}")
            if name.lower().startswith("p100") or cap[0] < 7:
                warn.append(("gpu", False, "P100/Volta 无 Tensor Core，Large@384 会很慢，先测 step 时间"))
            x = torch.randn(1024, 1024, device="cuda")
            (x @ x).sum().item()   # 有的镜像 is_available=True 但没编译当前 arch，这里才暴露
            print("[gpu] 1024x1024 matmul OK")
        else:
            ok.append(("gpu", False, "CUDA 不可用"))
        import importlib
        for m in ("timm", "sklearn", "pandas", "torchvision"):
            try:
                mod = importlib.import_module(m)
                print(f"[dep] {m} {getattr(mod, '__version__', '?')}")
            except ImportError:
                ok.append(("dep", False, f"{m} 未安装"))
    except ImportError:
        warn.append(("torch", False, "torch 未安装（在 Kaggle 上应先装好再开 GPU 会话）"))

    # 5) 续跑文件
    for cand in glob.glob("/kaggle/input/**/resume.pt", recursive=True):
        sz = os.path.getsize(cand) / 2 ** 30
        print(f"[resume] {cand}  {sz:.2f} GB")

    print("-" * 68)
    bad = [x for x in ok if not x[1]]
    for k, good, msg in ok:
        if not good:
            print(f"  FAIL {k}: {msg}")
    for _, _, msg in warn:
        print(f"  WARN {msg}")
    if not bad:
        print("  ALL CHECKS PASSED")

    print("-" * 68)
    print("按本机实际路径可直接粘贴：")
    print(f'  CACHE={cache_for_cmd}')
    print(f'  cd {os.path.abspath(args.repo)} && python train.py --arch convnext_large --img-size 384 \\')
    print(f'    --cache-root "$CACHE" --train-on-all --epochs 16 --batch-size 4 --accum 12 \\')
    print(f'    --lr 5e-5 --head-lr 2.5e-4 --pretrained 1 --randaugment 2 9 --seed 7 \\')
    print(f'    --loader workers --workers 2 --save-every-steps 200 \\')
    print(f'    --out-dir {args.work_dir}/runs/cnxl_384_e16_ra_all_s7 --tag cnxl_384_e16_ra_all_s7')
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
