"""参数选择: (β, α) 网格 + 数据项有无对比 (文献[3] 建议去掉数据项)."""
import sys, os, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from noise_model import add_salt_and_pepper, load_gray
from amf import adaptive_median_filter
from restoration_model import Phase2Model
from solvers import gpsr_bb
from metrics import psnr

IMG = os.path.join(ROOT, "data", "test_images", "lena_gray_512.tif")


u0_global = None  # main 中设为 AMF 输出 (solve 引用)


def solve(y, cand, beta, alpha, x, mu=1.0, with_data=True, maxit=1500):
    m = Phase2Model(y, cand, beta=beta, alpha=alpha, smooth="sqrt")
    m.with_data = with_data
    t0 = time.perf_counter()
    res = gpsr_bb(m, u0_global, mu=mu, tolP=1e-2, maxit=maxit)
    xh = y.copy(); xh[cand] = res["u"]
    return dict(it=res["it"], t=time.perf_counter() - t0, conv=res["converged"],
                psnr=psnr(x, xh))


def main():
    global u0_global
    x = load_gray(IMG)
    lines = ["# 问题一 (beta, alpha) 网格扫描 + 数据项有无对比",
             "# 当前两条件候选集管线; Lena 512, 种子 2026; mu=1, tolP=1e-2, maxit=1500"]
    for r in [0.3, 0.5]:
        rng = np.random.default_rng(2026)
        y, _ = add_salt_and_pepper(x, p=r / 2, q=r / 2, rng=rng)
        cand, y_amf, _ = adaptive_median_filter(y, r=r)
        u0_global = y_amf[cand]
        lines.append(f"\n===== r={r:.0%}, #cand={cand.sum()} =====")
        lines.append("  beta    alpha  withData     it     time    PSNR   conv")
        for beta in [5.0, 20.0, 40.0, 80.0, 200.0]:
            for alpha in [100.0, 300.0, 1000.0]:
                for wd in [True, False]:
                    r_ = solve(y, cand, beta, alpha, x, with_data=wd)
                    lines.append(
                        f"  {beta:6.1f} {alpha:7.1f}   {str(wd):5s} "
                        f"{r_['it']:6d} {r_['t']:7.2f} {r_['psnr']:8.3f}   {r_['conv']}")
                    print(lines[-1], flush=True)
    out = os.path.join(ROOT, "exp", "problem1", "results", "tables",
                       "param_grid_log.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("WROTE", out)


if __name__ == "__main__":
    main()
