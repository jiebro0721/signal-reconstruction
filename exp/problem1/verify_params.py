"""在多个图像上验证候选参数配置 (β=40, α=300) 与参考配置 (β=5, α=100)."""
import sys, os, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from noise_model import add_salt_and_pepper, load_gray
from amf import adaptive_median_filter
from restoration_model import Phase2Model
from solvers import gpsr_bb
from metrics import psnr

D = os.path.join(ROOT, "data", "test_images")
IMGS = ["cameraman.tif", "peppers_gray.tif", "lake.tif", "woman_blonde.tif"]


def solve(y, cand, x, beta, alpha, mu=1.0, maxit=1500):
    m = Phase2Model(y, cand, beta=beta, alpha=alpha, smooth="sqrt")
    t0 = time.perf_counter()
    res = gpsr_bb(m, y_amf[cand], mu=mu, tolP=1e-2, maxit=maxit)
    xh = y.copy(); xh[cand] = res["u"]
    return dict(it=res["it"], t=time.perf_counter() - t0, psnr=psnr(x, xh))


def main():
    for img in IMGS:
        x = load_gray(os.path.join(D, img))
        for r, beta, alpha in [(0.3, 40, 300), (0.3, 5, 100), (0.5, 40, 300), (0.5, 5, 100)]:
            rng = np.random.default_rng(2026)
            y, _ = add_salt_and_pepper(x, p=r / 2, q=r / 2, rng=rng)
            cand, y_amf, _ = adaptive_median_filter(y, r=r)
            res = solve(y, cand, x, beta, alpha)
            print(f"{img:18s} r={r:.0%} beta={beta:4.0f} alpha={alpha:4.0f} "
                  f"#cand={cand.sum():7d} it={res['it']:5d} t={res['t']:6.2f}s PSNR={res['psnr']:.3f}")


if __name__ == "__main__":
    main()
