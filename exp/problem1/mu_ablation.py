"""问题一 消融实验: 光滑参数 μ 对收敛效率与恢复质量的影响 (落盘存档).

依据审阅意见 M5: 论文 §5.7/§8.1 "μ<=1e-2 时上千步仍不收敛, 而恢复 PSNR 差
小于 0.01 dB" 的论断此前来自 tune.py 屏幕输出, 未写入 results. 本脚本在
Lena 30% (种子 2026) 上对比 μ ∈ {1, 1e-1, 1e-2}, maxit=3000:
  - μ=1:  数十步收敛;
  - μ=1e-2: 3000 步内不收敛 (gap 未达 tolP), 但 PSNR 与 μ=1 几乎相同.
协议与 run_problem1.py / make_figures.py 一致 (alpha=300, beta=40, tolP=1e-2).
输出: results/tables/problem1_mu_ablation.csv
"""
import sys, os, time, csv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from noise_model import add_salt_and_pepper, load_gray
from amf import adaptive_median_filter
from restoration_model import Phase2Model
from solvers import gpsr_bb
from metrics import psnr

IMG = os.path.join(ROOT, "data", "test_images", "lena_gray_512.tif")
SEED, R = 2026, 0.3
ALPHA, BETA = 300.0, 40.0
TOLP, MAXIT = 1e-2, 3000
MUS = (1.0, 1e-1, 1e-2, 1e-3, 1e-4)
TAB = os.path.join(ROOT, "exp", "problem1", "results", "tables")


def main():
    x = load_gray(IMG)
    rng = np.random.default_rng(SEED)
    y, _ = add_salt_and_pepper(x, p=R / 2, q=R / 2, rng=rng)
    cand, y_amf, _ = adaptive_median_filter(y, r=R)
    rows = []
    for mu in MUS:
        m = Phase2Model(y, cand, beta=BETA, alpha=ALPHA, smooth="sqrt")
        t0 = time.perf_counter()
        res = gpsr_bb(m, y_amf[cand], mu=mu, tolP=TOLP, maxit=MAXIT)
        t = time.perf_counter() - t0
        xh = y.copy(); xh[cand] = res["u"]
        rows.append(dict(mu=mu, it=int(res["it"]), t=round(t, 2),
                         converged=bool(res["converged"]),
                         final_gap=float(res["hist_gap"][-1]),
                         psnr=round(psnr(x, xh), 4)))
        print(rows[-1], flush=True)
    with open(os.path.join(TAB, "problem1_mu_ablation.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("DONE")


if __name__ == "__main__":
    main()
