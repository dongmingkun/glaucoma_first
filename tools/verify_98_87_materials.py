# -*- coding: utf-8 -*-
"""一条命令核验 98.87 材料包：成员 CSV 哈希 + 重算融合 MD5/SHA-256。

不联网、不读 test 标签；只比对仓库内文件与 manifest 记录的哈希，
再用 tools/reproduce_98_87.py 的同一套 rank 融合重算目标 CSV。

用法：
  python tools/verify_98_87_materials.py
退出码 0 = 全部 PASS；1 = 有 FAIL。
"""
import csv
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import reproduce_98_87 as repro  # noqa: E402

MANIFEST = os.path.join(ROOT, "docs", "audit_98_87", "manifest_98_87.json")


def hashes(b):
    lf = b.replace(b"\r\n", b"\n")
    return {
        "md5": hashlib.md5(b).hexdigest(),
        "sha256": hashlib.sha256(b).hexdigest(),
        "md5_lf_normalized": hashlib.md5(lf).hexdigest(),
        "sha256_lf_normalized": hashlib.sha256(lf).hexdigest(),
    }


def main():
    if not os.path.isfile(MANIFEST):
        sys.exit("ERROR: 缺 %s（先生成 manifest）" % MANIFEST)
    m = json.load(open(MANIFEST, encoding="utf-8"))
    target = m["target"]
    fails = []

    def report(ok, msg):
        print(("PASS  " if ok else "FAIL  ") + msg)
        if not ok:
            fails.append(msg)

    print("target: md5=%s sha256=%s rows=%d"
          % (target["csv_md5"], target["csv_sha256"], target["rows"]))

    for key, e in sorted(m["csv_files"].items()):
        path = os.path.join(ROOT, e["file"])
        if not os.path.isfile(path):
            report(False, "%s: 缺文件 %s" % (key, e["file"]))
            continue
        b = open(path, "rb").read()
        h = hashes(b)
        rows = sum(1 for _ in csv.DictReader(open(path, encoding="utf-8", newline="")))
        ok = (len(b) == e["bytes"] and rows == e["rows"]
              and h["md5"] == e["md5"] and h["sha256"] == e["sha256"]
              and h["md5_lf_normalized"] == e["md5_lf_normalized"]
              and h["sha256_lf_normalized"] == e["sha256_lf_normalized"])
        report(ok, "%s: %s (%d B, %d 行)" % (key, e["file"], len(b), rows))
        if not ok:
            print("      got   md5=%s sha256=%s" % (h["md5"], h["sha256"]))
            print("      want  md5=%s sha256=%s" % (e["md5"], e["sha256"]))

    test_csv = repro.DEFAULTS["test_csv"]
    if not os.path.isfile(test_csv):
        report(False, "官方 test.csv 不在仓库内（%s），跳过重算" % test_csv)
    else:
        with open(test_csv, encoding="utf-8", newline="") as f:
            names = [r["Image_name"] for r in csv.DictReader(f)]
        score = repro.build(names, repro.DEFAULTS)
        blob = repro.to_csv_bytes(names, score)
        h = hashes(blob)
        ok = (len(names) == target["rows"]
              and h["md5"] == target["csv_md5"]
              and h["sha256"] == target["csv_sha256"])
        report(ok, "reproduce: md5=%s sha256=%s" % (h["md5"], h["sha256"]))

    pkg = m["package"]
    print("\nweights: published %d B (%.1f MiB) in %d assets; "
          "Swin checkpoints unpublished (estimate +%d B); total %.2f GB"
          % (pkg["members_published_weight_bytes"],
             pkg["members_published_weight_bytes"] / 2 ** 20,
             sum(1 for x in m["models"] if x.get("weights_asset")),
             pkg["members_swin_estimate_bytes"],
             pkg["members_total_with_estimate_bytes"] / 1e9))
    print("compliant_98_87: %s" % pkg["compliant_98_87"])

    if fails:
        print("\n%d FAIL" % len(fails))
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
