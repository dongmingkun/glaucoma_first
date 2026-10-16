"""Download the RETFound (MAE, retinal foundation model) checkpoint from HuggingFace.

The repo is gated: log into HuggingFace, open
  https://huggingface.co/YukunZhou/RETFound_mae_natureCFP
and click "Agree and access repository". Then put a READ token in
  D:\\glaucoma_work\\.hf_token      (or set HF_TOKEN env var)

Usage:
  python tools/download_retfound.py                 # 默认下载 natureCFP 版本
  python tools/download_retfound.py --repo YukunZhou/RETFound_mae_natureOCT
"""
import argparse
import os
import subprocess
import sys

DEST_DIR = r"D:\glaucoma_work\weights"
PROXY = os.environ.get("GLAUCOMA_PROXY", "http://127.0.0.1:7897")
TOKEN_FILE = r"D:\glaucoma_work\.hf_token"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="YukunZhou/RETFound_mae_natureCFP")
    ap.add_argument("--file", default="RETFound_mae_natureCFP.pth")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and os.path.isfile(TOKEN_FILE):
        token = open(TOKEN_FILE).read().strip()
    if not token:
        print(f"!! 未找到 HF token：请把 read 权限的 token 写入 {TOKEN_FILE}")
        sys.exit(1)

    os.makedirs(DEST_DIR, exist_ok=True)
    out = args.out or os.path.join(DEST_DIR, args.file)
    url = f"https://huggingface.co/{args.repo}/resolve/main/{args.file}"
    print(f"下载 {url}\n  -> {out}")

    cmd = ["curl", "-L", "-C", "-", "--retry", "5", "--retry-all-errors",
           "--connect-timeout", "30", "--max-time", "5400",
           "-H", f"Authorization: Bearer {token}",
           "-o", out, url]
    if PROXY:
        cmd = cmd[:1] + ["-x", PROXY] + cmd[1:]
    r = subprocess.run(cmd, text=True)
    if r.returncode != 0:
        print(f"!! 下载失败 (curl {r.returncode})；若是 401/403，请确认已在网页上接受条款")
        sys.exit(r.returncode)
    sz = os.path.getsize(out)
    print(f"完成：{out}  ({sz/1e6:.1f} MB)")
    if sz < 100e6:
        print("!! 文件过小，可能下到的是错误页而不是权重")


if __name__ == "__main__":
    main()