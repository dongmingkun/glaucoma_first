"""Assemble a degraded submission package from the existing Release weights.

The two ConvNeXt-Base@384 weights already live in Release `weights-1016`; this
script only re-packages a *subset* into a self-contained, <=500 MB folder.

  - cb384.pt              : already a slim single-state fp16 weight (168 MB)
  - cnxb_384_all_pseudo.pt: 336 MB double-state fp16 in the Release -> slimmed to
                           single-state fp16 (176 MB), matching the 3-model package

The cnxb slim needs torch. If torch is unavailable the script downloads the
pre-slimmed asset from the companion release instead (URL printed).

Usage (run from inside the package directory, e.g. submission_pkg_1model/):
  python assemble.py --mode 1model     # downloads + slims cnxb_384_all_pseudo.pt
  python assemble.py --mode 2model     # also downloads cb384.pt
"""
import argparse
import hashlib
import os
import subprocess
import sys

REPO = "dongmingkun/glaucoma_first"
RELEASE = "weights-1016"
CB384_SRC = "convnext_base_384_cache512_1015_auprc0.8644.pt"   # slim single-state
CNXB_SRC = "cnxb_384_all_pseudo_fp16.pt"                       # double-state -> slim


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True)


def download(asset, dst):
    url = f"https://github.com/{REPO}/releases/download/{RELEASE}/{asset}"
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  skip {dst} (exists)", flush=True)
        return
    # curl is universal (Win10+, Linux, macOS) and resumable.
    if sys.platform.startswith("win"):
        run(["curl", "-L", "-C", "-", "-o", dst, url])
    else:
        run(["curl", "-L", "--retry", "3", "-o", dst, url])


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slim_cnxb(src, dst):
    """Slim the 336 MB double-state fp16 ckpt to a single-state fp16 inference weight.

    Mirrors tools/slim_weights.py (use=ema, matching the 3-model package)."""
    import torch
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    use = "ema"  # train+all+pseudo weight uses ema in the 3-model package
    sd = ckpt.get(use) or ckpt.get("model") or ckpt
    arch = ckpt.get("arch", "convnext_base")
    img_size = ckpt.get("img_size", 384)
    src_args = ckpt.get("args") if isinstance(ckpt, dict) else None
    src_cache = str(src_args.get("cache_root", "")) if isinstance(src_args, dict) else ""
    cache_size = 768 if "768" in (src_cache + os.path.basename(dst)) else 512
    out_sd = {}
    for k, v in sd.items():
        if v.is_floating_point() and v.dtype == torch.float32 and not (
            k.endswith("running_mean") or k.endswith("running_var")
        ):
            out_sd[k] = v.half()
        else:
            out_sd[k] = v
    out_ckpt = {
        "state_dict": out_sd,
        use: out_sd,
        "arch": arch,
        "img_size": img_size,
        "use": use,
        "fp16": True,
        "cache_size": cache_size,
        "args": {"cache_root": src_cache or f"cache{cache_size}"},
        "source": os.path.basename(src),
    }
    torch.save(out_ckpt, dst)
    print(f"  slimmed -> {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["1model", "2model"])
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    download(CNXB_SRC, os.path.join(here, "_cnxb_raw.pt"))
    try:
        slim_cnxb(os.path.join(here, "_cnxb_raw.pt"),
                   os.path.join(here, "cnxb_384_all_pseudo.pt"))
    except ImportError:
        print("torch not found; download the pre-slimmed asset from the "
              "companion release `submission-degraded-110`:", flush=True)
        print(f"  https://github.com/{REPO}/releases/tag/submission-degraded-110",
              flush=True)
        sys.exit(2)
    finally:
        rp = os.path.join(here, "_cnxb_raw.pt")
        if os.path.exists(rp):
            os.remove(rp)

    if args.mode == "2model":
        download(CB384_SRC, os.path.join(here, "cb384.pt"))

    for f in sorted(os.listdir(here)):
        if f.endswith(".pt"):
            p = os.path.join(here, f)
            print(f"  {f}: {os.path.getsize(p)/1e6:.1f} MB  md5={md5(p)[:8]}", flush=True)
    print("assemble done. Run `python measure_latency.py --help` for the timing command.",
          flush=True)


if __name__ == "__main__":
    main()
