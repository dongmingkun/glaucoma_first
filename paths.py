"""Shared path resolution (no torch dependency).

Every script used to hard-code the captain's machine paths, which broke
--train-on-all (and everything else) on any other machine. Resolution order:

  1. env var  GLAUCOMA_CSV_ROOT   / GLAUCOMA_CACHE_ROOT
  2. repo layout  <repo>/glaucoma_first/{train,valid,test}.csv
  3. legacy hard-coded paths (captain's machine)

Import this instead of re-declaring the constants so lightweight scripts
(verify_cache.py) don't have to pull in torch just to find the data.
"""
import os

AGE_MEDIAN = 58.0

_HERE = os.path.dirname(os.path.abspath(__file__))


def _first_existing(paths):
    for p in paths:
        if p and os.path.isdir(p):
            return p
    return None


def _detect_csv_root():
    """Folder holding train.csv / valid.csv / test.csv."""
    found = _first_existing([
        os.environ.get("GLAUCOMA_CSV_ROOT"),
        os.path.join(_HERE, "glaucoma_first"),
        os.path.join(os.path.dirname(_HERE), "glaucoma_first"),
        os.path.join(_HERE, "glaucoma_first", "glaucoma_first"),
        r"C:\Users\Kevin\Desktop\glaucoma_first\glaucoma_first",
    ])
    return found or os.path.join(_HERE, "glaucoma_first")


def _detect_cache_root():
    """Folder holding the 512px cache produced by preprocess.py."""
    found = _first_existing([
        os.environ.get("GLAUCOMA_CACHE_ROOT"),
        os.environ.get("GLAUCOMA_CACHE"),   # 兼容旧文档中的变量名
        r"D:\glaucoma_work\cache512",
        os.path.join(os.path.dirname(_HERE), "cache512"),
        os.path.join(_HERE, "cache512"),
    ])
    return found or r"D:\glaucoma_work\cache512"


CACHE_ROOT = _detect_cache_root()
CSV_ROOT = _detect_csv_root()


def load_rows(split, csv_root=None):
    """Read split CSV -> list of (stem, cleaned_age, label). label=-1 if absent."""
    import csv

    root = csv_root or CSV_ROOT
    path = os.path.join(root, split + ".csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found (CSV_ROOT={root}).\n"
            f"  -> pass --csv-root <dir> or set env GLAUCOMA_CSV_ROOT to the folder "
            f"holding train.csv / valid.csv / test.csv")
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            name = r["Image_name"]
            stem = name.rsplit(".", 1)[0]
            try:
                age = float(r["Age"])
            except (ValueError, TypeError):
                age = AGE_MEDIAN
            if age < 1 or age > 110:
                age = AGE_MEDIAN
            label = float(r["True_Label"]) if "True_Label" in r else -1.0
            rows.append((stem, min(max(age, 10.0), 100.0), label))
    return rows
