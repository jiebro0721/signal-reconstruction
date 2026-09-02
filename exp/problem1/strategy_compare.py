"""策略对比实验: 求解器(Basic/BB1/BB2), 光滑参数(固定μ/续延), 数据项有无."""
import sys, os, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from noise_model import add_salt_and_pepper, load_gray
from amf import adaptive_median_filter
from restoration_model import Phase2Model
from solvers import gpsr_bb, gpsr_basic, solve_with_continuation
from metrics import psnr, snr

D = os.path.join(ROOT, "data", "test_images")
TAB = os.path.join(ROOT, "exp", "problem1", "results", "tables")
IMGS = ["lena_gray_512.tif", "peppers_gray.tif"]
BETA, ALPHA = 40.0, 300.0


def base(y, cand, x):
    m = Phase2Model(y, cand, beta=BETA, alpha=ALPHA, smooth="sqrt")
    return m, y[cand]


def summary(res, y, cand, x):
    xh = y.copy(); xh[cand] = res["u"]
    return f"it={res['it'] if 'it' in res else res['it_sum']:5d} " \
           f"t={res['time'] if 'time' in res else float('nan'):6.2f}s " \
           f"PSNR={psnr(x, xh):7.3f} SNR={snr(x, xh):7.3f} conv={res['converged']}"


def main():
    lines = []
    for img in IMGS:
        x = load_gray(os.path.join(D, img))
        for r in (0.3, 0.5):
            rng = np.random.default_rng(2026)
            y, _ = add_salt_and_pepper(x, p=r / 2, q=r / 2, rng=rng)
            cand, y_amf, _ = adaptive_median_filter(y, r=r)
            lines.append(f"\n== {img} r={r:.0%} #cand={cand.sum()} ==")
            # 1) 求解器对比 (μ=1, 含数据项)
            m, u0 = base(y, cand, x)
            t0 = time.perf_counter()
            res = gpsr_basic(m, u0, mu=1.0, tolP=1e-2, maxit=1500)
            res["time"] = time.perf_counter() - t0
            lines.append(f"  [Basic]    {summary(res, y, cand, x)}")
            m, u0 = base(y, cand, x)
            t0 = time.perf_counter()
            res = gpsr_bb(m, u0, mu=1.0, tolP=1e-2, maxit=1500)
            res["time"] = time.perf_counter() - t0
            lines.append(f"  [BB1]      {summary(res, y, cand, x)}")
            m, u0 = base(y, cand, x)
            t0 = time.perf_counter()
            res = gpsr_bb(m, u0, mu=1.0, tolP=1e-2, maxit=1500, bb_variant=2)
            res["time"] = time.perf_counter() - t0
            lines.append(f"  [BB2]      {summary(res, y, cand, x)}")
            # 2) 光滑参数策略 (BB1)
            m, u0 = base(y, cand, x)
            t0 = time.perf_counter()
            res = solve_with_continuation(m, u0, mu_seq=(1.0, 1e-1, 1e-2, 1e-3),
                                          tolP=1e-2, maxit_each=1500)
            res["time"] = time.perf_counter() - t0
            lines.append(f"  [续延μ]    {summary(res, y, cand, x)}")
            # 3) 数据项有无 (BB1, μ=1)
            m, u0 = base(y, cand, x)
            m.with_data = False
            t0 = time.perf_counter()
            res = gpsr_bb(m, u0, mu=1.0, tolP=1e-2, maxit=1500)
            res["time"] = time.perf_counter() - t0
            lines.append(f"  [无数据项] {summary(res, y, cand, x)}")
    out = "\n".join(lines)
    print(out)
    with open(os.path.join(TAB, "problem1_strategy.txt"), "w", encoding="utf-8") as f:
        f.write(out)


if __name__ == "__main__":
    main()
