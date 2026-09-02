"""从 problem1_full.csv 重新生成正确的均值汇总表 (修复列错位)."""
import csv, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TAB = os.path.join(ROOT, "exp", "problem1", "results", "tables")
rows = list(csv.DictReader(open(os.path.join(TAB, "problem1_full.csv"), encoding="utf-8-sig")))
keys = ["it", "t_amf", "t_solve", "psnr", "snr", "mae", "psnr_n", "tpr", "fpr"]
groups = {}
for q in rows:
    groups.setdefault((q["image"], q["r"]), []).append(q)
with open(os.path.join(TAB, "problem1_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["image", "r", "n_cand", "wmax"] + [f"{k}_mean" for k in keys] +
               [f"{k}_std" for k in keys])
    for (img, r), sub in groups.items():
        vals = {k: [float(q[k]) for q in sub] for k in keys}
        w.writerow([img, r, sub[0]["n_cand"], sub[0]["wmax"]] +
                   [float(np.mean(vals[k])) for k in keys] +
                   [float(np.std(vals[k])) for k in keys])
print("summary regenerated")
