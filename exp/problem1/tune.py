"""求解器与参数快速调优: 光滑参数 μ、续延、BB 与 Basic、β 敏感性."""
import sys, os, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from noise_model import add_salt_and_pepper, load_gray
from amf import adaptive_median_filter
from restoration_model import Phase2Model
from solvers import gpsr_bb, gpsr_basic, solve_with_continuation
from metrics import psnr

IMG = os.path.join(ROOT, "data", "test_images", "lena_gray_512.tif")


def run_one(x, y, cand, beta, mu=1e-3, solver="bb", maxit=4000):
    model = Phase2Model(y, cand, beta=beta, alpha=100.0, smooth="sqrt")
    u0 = y_amf[cand]
    t0 = time.perf_counter()
    if solver == "bb":
        res = gpsr_bb(model, u0, mu=mu, tolP=1e-2, maxit=maxit)
    elif solver == "basic":
        res = gpsr_basic(model, u0, mu=mu, tolP=1e-2, maxit=maxit)
    elif solver == "cont":
        res = solve_with_continuation(model, u0, mu_seq=(1e-1, 1e-2, 1e-3),
                                      tolP=1e-2, maxit_each=maxit)
    xh = y.copy(); xh[cand] = res["u"]
    dt = time.perf_counter() - t0
    it = res["it_sum"] if "it_sum" in res else res["it"]
    return dict(solver=solver, beta=beta, mu=mu, it=it,
                time=dt, psnr=psnr(x, xh), conv=res["converged"],
                gap=res["hist_gap"][-1] if "hist_gap" in res else np.nan)


def main():
    x = load_gray(IMG)
    print(f"Lena 512, r=30%")
    rng = np.random.default_rng(2026)
    y, true_mask = add_salt_and_pepper(x, p=0.15, q=0.15, rng=rng)
    cand, y_amf, _ = adaptive_median_filter(y, r=0.3)

    print("\n== μ 影响 (BB, beta=5) ==")
    for mu in [1.0, 1e-1, 1e-2, 1e-3]:
        r = run_one(x, y, cand, 5.0, mu=mu, solver="bb", maxit=2000)
        print(f"  mu={mu:g}: it={r['it']:5d} t={r['time']:6.1f}s PSNR={r['psnr']:.3f} "
              f"conv={r['conv']} gap={r['gap']:.2e}")

    print("\n== BB vs Basic vs 续延 (beta=5) ==")
    for solver in ["bb", "basic", "cont"]:
        r = run_one(x, y, cand, 5.0, solver=solver, maxit=2000)
        print(f"  {solver:5s}: mu={r['mu']:g} it={r['it']:5d} t={r['time']:6.1f}s PSNR={r['psnr']:.3f} "
              f"conv={r['conv']} gap={r['gap']:.2e}")

    print("\n== 续延参数序列 (beta=5) ==")
    for seq in [(1.0,), (1.0, 1e-1), (1.0, 1e-1, 1e-2), (1.0, 1e-1, 1e-2, 1e-3)]:
        model = Phase2Model(y, cand, beta=5.0, alpha=100.0, smooth="sqrt")
        u0 = y_amf[cand]
        t0 = time.perf_counter()
        res = solve_with_continuation(model, u0, mu_seq=seq, tolP=1e-2, maxit_each=2000)
        xh = y.copy(); xh[cand] = res["u"]
        print(f"  {seq}: it={res['it_sum']:5d} each={res['it_hist']} "
              f"t={time.perf_counter()-t0:6.1f}s PSNR={psnr(x, xh):.3f} conv={res['converged']}")

    print("\n== β 敏感性 (BB, mu=1) ==")
    for beta in [1.0, 2.0, 5.0, 10.0, 20.0, 40.0]:
        r = run_one(x, y, cand, beta, mu=1.0, solver="bb", maxit=1500)
        print(f"  beta={beta:5.1f}: it={r['it']:5d} t={r['time']:6.1f}s PSNR={r['psnr']:.3f} "
              f"conv={r['conv']} gap={r['gap']:.2e}")

    print("\n== α 敏感性 (BB, mu=1, beta=5) ==")
    for alpha in [30.0, 100.0, 300.0]:
        model = Phase2Model(y, cand, beta=5.0, alpha=alpha, smooth="sqrt")
        u0 = y_amf[cand]
        t0 = time.perf_counter()
        res = gpsr_bb(model, u0, mu=1.0, tolP=1e-2, maxit=1500)
        xh = y.copy(); xh[cand] = res["u"]
        print(f"  alpha={alpha:6.1f}: it={res['it']:5d} t={time.perf_counter()-t0:6.1f}s "
              f"PSNR={psnr(x, xh):.3f} conv={res['converged']} gap={res['hist_gap'][-1]:.2e}")

    print("\n== r=50% 检查 (BB, mu=1) ==")
    rng50 = np.random.default_rng(2026)
    y50, _ = add_salt_and_pepper(x, p=0.25, q=0.25, rng=rng50)
    cand50, y50_amf, _ = adaptive_median_filter(y50, r=0.5)
    print(f"  #cand={cand50.sum()}")
    for beta in [1.0, 5.0, 10.0, 20.0]:
        model = Phase2Model(y50, cand50, beta=beta, alpha=100.0, smooth="sqrt")
        t0 = time.perf_counter()
        res = gpsr_bb(model, y50_amf[cand50], mu=1.0, tolP=1e-2, maxit=1500)
        xh = y50.copy(); xh[cand50] = res["u"]
        print(f"  beta={beta:5.1f}: it={res['it']:5d} t={time.perf_counter()-t0:6.1f}s "
              f"PSNR={psnr(x, xh):.3f} conv={res['converged']} gap={res['hist_gap'][-1]:.2e}")


if __name__ == "__main__":
    main()
