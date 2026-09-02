"""单元验证: 噪声模型 / AMF 检测 / 光滑模型梯度 / 求解器收敛."""
import sys, os, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from noise_model import add_salt_and_pepper, load_gray
from amf import adaptive_median_filter, w_max_for_noise_level
from restoration_model import Phase2Model, finite_diff_check
from solvers import gpsr_bb, gpsr_basic, solve_with_continuation
from metrics import psnr, snr, detection_stats

IMG = os.path.join(ROOT, "data", "test_images", "lena_gray_512.tif")

def main():
    x = load_gray(IMG)
    rng = np.random.default_rng(2026)
    r = 0.3
    y, true_mask = add_salt_and_pepper(x, p=r / 2, q=r / 2, rng=rng)
    print(f"image {x.shape}, noise level r={r}, true noisy pixels: {true_mask.sum()} "
          f"({100*true_mask.mean():.2f}%)")

    # ---- AMF ----
    wmax = w_max_for_noise_level(r)
    t0 = time.perf_counter()
    cand, y_amf, _ = adaptive_median_filter(y, r=r)
    print(f"AMF wmax={wmax} time={time.perf_counter()-t0:.2f}s "
          f"detection: {detection_stats(true_mask, cand)}")
    print(f"  PSNR(y->y_amf) = {psnr(x, y_amf):.2f} dB, PSNR(y) = {psnr(x, y):.2f} dB")

    # ---- 模型 ----
    model = Phase2Model(y, cand, beta=5.0, alpha=100.0, smooth="sqrt")
    u0 = y_amf[cand]
    print(f"#vars = {model.nvars}")
    for mu in [1e-1, 1e-2, 1e-3]:
        err = finite_diff_check(model, u0, mu, eps=1e-5)
        print(f"  FD gradient check (mu={mu}): max rel err = {err:.3e}")

    # ---- 求解 ----
    for beta in [1.0, 5.0, 10.0]:
        model.beta = beta
        t0 = time.perf_counter()
        res = gpsr_bb(model, u0, mu=1e-3, tolP=1e-2, maxit=3000)
        xh = y.copy(); xh[cand] = res["u"]
        print(f"  BB  beta={beta}: it={res['it']} time={time.perf_counter()-t0:.2f}s "
              f"conv={res['converged']} PSNR={psnr(x, xh):.3f} dB SNR={snr(x, xh):.3f} dB "
              f"gap_end={res['hist_gap'][-1]:.2e}")

    model.beta = 5.0
    t0 = time.perf_counter()
    res = gpsr_basic(model, u0, mu=1e-3, tolP=1e-2, maxit=3000)
    xh = y.copy(); xh[cand] = res["u"]
    print(f"Basic beta=5: it={res['it']} time={time.perf_counter()-t0:.2f}s "
          f"conv={res['converged']} PSNR={psnr(x, xh):.3f} dB gap_end={res['hist_gap'][-1]:.2e}")

    # ---- 续延 ----
    t0 = time.perf_counter()
    res = solve_with_continuation(model, u0, mu_seq=(1e-1, 1e-2, 1e-3), tolP=1e-2, maxit_each=3000)
    xh = y.copy(); xh[cand] = res["u"]
    print(f"continuation: it_sum={res['it_sum']} each={res['it_hist']} "
          f"time={time.perf_counter()-t0:.2f}s PSNR={psnr(x, xh):.3f} dB")

if __name__ == "__main__":
    main()
