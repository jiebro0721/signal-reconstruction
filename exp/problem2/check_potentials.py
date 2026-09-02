"""问题二: 5 种势函数的梯度有限差分校验 + 凸性/KKT 简单检查 (Lena 512, 30%)."""
import sys, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from noise_model import add_salt_and_pepper, load_gray
from amf import adaptive_median_filter
from restoration_model import Phase2Model, finite_diff_check
from solvers import gpsr_bb
from metrics import psnr

x = load_gray(os.path.join(ROOT, "data", "test_images", "lena_gray_512.tif"))
rng = np.random.default_rng(2026)
y, _ = add_salt_and_pepper(x, p=0.15, q=0.15, rng=rng)
cand, y_amf, _ = adaptive_median_filter(y, r=0.3)

for pot_, alpha in [("sqrt", 300.0), ("power", 1.4), ("logcosh", 0.1),
                    ("log1", 3.0), ("huber", 30.0)]:
    m = Phase2Model(y, cand, beta=40.0, alpha=alpha, smooth="sqrt", potential=pot_)
    err = finite_diff_check(m, y_amf[cand], mu=1.0, eps=1e-5, n_sample=20)
    res = gpsr_bb(m, y_amf[cand], mu=1.0, tolP=1e-2, maxit=1500)
    xh = y.copy(); xh[cand] = res["u"]
    # 凸性抽查: 中线凸性 f(0.5(u+v)) <= 0.5(f(u)+f(v))
    rng2 = np.random.default_rng(1)
    u = y_amf[cand].copy()
    v = res["u"].copy()
    mid = 0.5 * (u + v)
    convex = m.value(mid) <= 0.5 * (m.value(u) + m.value(v)) + 1e-6 * abs(m.value(u))
    print(f"{pot_:8s} FD_err={err:.2e} it={res['it']:4d} conv={res['converged']} "
          f"PSNR={psnr(x, xh):.3f} 中间凸性抽查={convex}")
