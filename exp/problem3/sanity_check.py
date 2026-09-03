"""问题三单元验证: 实例生成 / 梯度 FD / PRP+CG 与 GPSR-BB 各自收敛性."""
import sys, os, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sparse_reconstruction import (make_instance, tau_gpsr, SmoothedL1Quad,
                                   finite_diff_check)
from cg_solvers import nonlinear_cg, cg_with_mu_continuation, gpsr_bb_solve

A, x, b = make_instance(seed=2026)
tau = tau_gpsr(A, b)
print(f"instance: n={len(x)} k={A.shape[0]} nz={np.count_nonzero(x)} "
      f"|x|2={np.linalg.norm(x):.2f} tau={tau:.4f}")

model = SmoothedL1Quad(A, b, tau, mu=1e-2)
x0 = np.zeros_like(x)
print("FD grad check:", finite_diff_check(model, x0 + 0.01, eps=1e-6))
# 有偏初值也验一次
print("FD grad check 2:", finite_diff_check(model, A.T @ b * 0.1, eps=1e-6))

t0 = time.perf_counter()
res = cg_with_mu_continuation(model, x0, mu_seq=(1e-1, 1e-2, 1e-3),
                              beta="prp+", maxit=4000, tolG=1e-6)
t_cg = time.perf_counter() - t0
rel = np.linalg.norm(res["x"] - x) / np.linalg.norm(x)
sup = np.sum(np.abs(res["x"]) > 0.05)
print(f"CG PRP+: it={res['it_sum']} each={res['it_hist']} conv={res['conv']} "
      f"t={t_cg:.2f}s rel_err={rel:.5f} |support|={sup}")

t0 = time.perf_counter()
g = gpsr_bb_solve(A, b, tau, tolP=1e-3, maxit=3000)
t_gpsr = time.perf_counter() - t0
relg = np.linalg.norm(g["x"] - x) / np.linalg.norm(x)
supg = np.sum(np.abs(g["x"]) > 0.05)
print(f"GPSR-BB: it={g['it']} conv={g['conv']} t={t_gpsr:.2f}s "
      f"rel_err={relg:.5f} |support|={supg}")

t0 = time.perf_counter()
gd = gpsr_bb_solve(A, b, tau, tolP=1e-3, maxit=3000, do_debias=True)
t_gpsr_d = time.perf_counter() - t0
reld = np.linalg.norm(gd["x"] - x) / np.linalg.norm(x)
print(f"GPSR-BB+debias: t={t_gpsr_d:.2f}s rel_err={reld:.5f}")

# 目标值对比 (在最终解上计算非光滑目标)
def obj(v, tau):
    return tau * np.sum(np.abs(v)) + 0.5 * np.sum((A @ v - b) ** 2)
print(f"obj CG={obj(res['x'], tau):.6f}  GPSR={obj(g['x'], tau):.6f}  "
      f"GPSR+db={obj(gd['x'], tau):.6f}  true(x)={obj(x, tau):.6f}")
