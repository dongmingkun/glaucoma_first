"""Package a submission candidate + optional final model package into one delivery zip.

Verifies the platform requirements before writing anything:
  * CSV: 5855 rows, header Image_name,Probability,Pred_Label, names match test.csv order,
    probabilities in [0,1], no NaN, Pred_Label consistent with p>=0.5.
  * Package (optional): predict.py + weights, total weights <= 500MB.

Output layout:
  D:/glaucoma_work/submissions/<date>_<name>/
    submission.csv
    manifest.md
    package/            (optional)
    <date>_<name>.zip

Usage:
  python package_submission.py --csv submission_20251018_ens10_cal.csv --name ens10 \
      [--include-package] [--expected 98.82] [--out-root D:/glaucoma_work/submissions]
"""
import argparse
import csv
import hashlib
import os
import shutil
import time
import zipfile

TEST_CSV = os.path.join("glaucoma_first", "test.csv")
PKG_DIR = "submit_pkg"
MAX_PKG_BYTES = 500 * 1000 * 1000
PKG_FILES = ["predict.py", "README.md", "cb384.pt", "cnxb_384_all_pseudo.pt", "cs_hi512_all_ra.pt"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_csv(path):
    names_ref = []
    with open(TEST_CSV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            names_ref.append(r["Image_name"])

    names, probs, labels = [], [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != ["Image_name", "Probability", "Pred_Label"]:
            raise SystemExit(f"bad header: {rd.fieldnames}")
        for r in rd:
            names.append(r["Image_name"])
            p = float(r["Probability"])
            probs.append(p)
            labels.append(int(r["Pred_Label"]))

    if len(names) != len(names_ref):
        raise SystemExit(f"row count {len(names)} != {len(names_ref)}")
    if names != names_ref:
        bad = next(i for i, (a, b) in enumerate(zip(names, names_ref)) if a != b)
        raise SystemExit(f"image order mismatch at row {bad}: {names[bad]} vs {names_ref[bad]}")
    if any(p != p or p < 0 or p > 1 for p in probs):
        raise SystemExit("probability out of [0,1] or NaN")
    mism = sum(1 for p, lab in zip(probs, labels) if (p >= 0.5) != bool(lab))
    if mism:
        raise SystemExit(f"Pred_Label inconsistent with p>=0.5 on {mism} rows")

    import numpy as np
    pos = float((np.array(probs) >= 0.5).mean())
    return len(names), pos


def verify_pkg(pkg_dir):
    missing = [f for f in PKG_FILES if not os.path.exists(os.path.join(pkg_dir, f))]
    if missing:
        raise SystemExit(f"package missing files: {missing}")
    total = sum(os.path.getsize(os.path.join(pkg_dir, f)) for f in PKG_FILES if f.endswith(".pt"))
    if total > MAX_PKG_BYTES:
        raise SystemExit(f"weights {total/1e6:.1f}MB > 500MB")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--expected", default=None, help="expected LB score, for the manifest")
    ap.add_argument("--include-package", action="store_true")
    ap.add_argument("--out-root", default=r"D:/glaucoma_work/submissions")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    # Naming rule: upload file must be exactly `submission.csv`; keep the dated
    # internal name (submission_<date>_<tag>_cal.csv) for traceability.
    stem = os.path.splitext(os.path.basename(args.csv))[0]
    if not stem.startswith("submission_"):
        stem = f"submission_{time.strftime('%Y%m%d')}_{args.name}_cal"
    out_dir = os.path.join(args.out_root, stem)
    os.makedirs(out_dir, exist_ok=True)

    rows, pos = verify_csv(args.csv)
    dst_csv = os.path.join(out_dir, "submission.csv")
    shutil.copyfile(args.csv, dst_csv)
    internal_csv = os.path.join(out_dir, stem + ".csv")
    shutil.copyfile(args.csv, internal_csv)
    pkg_mb = None
    if args.include_package:
        pkg_mb = verify_pkg(PKG_DIR)
        pkg_out = os.path.join(out_dir, "package")
        if os.path.isdir(pkg_out):
            shutil.rmtree(pkg_out)
        os.makedirs(pkg_out)
        for f in PKG_FILES:
            shutil.copy2(os.path.join(PKG_DIR, f), pkg_out)

    lines = [
        f"# Submission bundle — {args.name}",
        "",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- upload file: `submission.csv` ({rows} rows, pos_rate={pos:.4f})",
        f"- internal name: `{stem}.csv` (same file, for traceability)",
        f"- csv md5: `{md5(dst_csv)}`",
    ]
    if args.expected:
        lines.append(f"- expected LB: **{args.expected}**")
    if pkg_mb is not None:
        lines += [
            f"- package: `package/` ({pkg_mb/1e6:.1f}MB weights, <=500MB)",
            "",
            "## Reproduce (final package)",
            "```bash",
            "python package/predict.py --images <test_images> --csv <test.csv> \\",
            "  --weights package/cb384.pt package/cnxb_384_all_pseudo.pt package/cs_hi512_all_ra.pt \\",
            "  --blend 0.5 0.25 0.25 --tta 1 --batch-size 32 --prep-workers 8 --out submission.csv",
            "```",
        ]
    with open(os.path.join(out_dir, "manifest.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    zip_path = None
    if not args.no_zip:
        zip_path = os.path.join(args.out_root, stem + ".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.write(dst_csv, "submission.csv")
            z.write(internal_csv, stem + ".csv")
            z.write(os.path.join(out_dir, "manifest.md"), "manifest.md")
            if args.include_package:
                for f in PKG_FILES:
                    z.write(os.path.join(PKG_DIR, f), f"package/{f}")
        print(f"zip: {zip_path} ({os.path.getsize(zip_path)/1e6:.1f}MB) md5={md5(zip_path)}")

    print(f"bundle: {out_dir}")
    print(f"upload: {dst_csv} md5={md5(dst_csv)} rows={rows} pos_rate={pos:.4f}")
    if pkg_mb is not None:
        print(f"package weights: {pkg_mb/1e6:.1f}MB (limit 500MB)")


if __name__ == "__main__":
    main()
