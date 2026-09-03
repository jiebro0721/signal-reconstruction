"""问题三 消融实验: 初值选择与分段停止准则对"同目标值口径"迭代数的影响.

依据审阅意见 M5: 论文 §7.5 "零初值 + 梯度准则需 257 步" 的论断此前为临时运行,
未落盘. 本脚本在 10 个实例上对比两种配置达到 GPSR-BB 目标值 ×1.001 的累计
迭代数/时间, 并落盘为 CSV:
  A. x0 = 0       + 各段以梯度无穷范数 <= tolG 停止 (梯度准则);
  B. x0 = Aᵀb     + 各段以相对目标变化 <= rel_tol 停止 (主实验配置, 文献[4]).
其余超参与 run_problem3.py 完全一致 (μ 序列, 强 Wolfe, PRP+, 下降性重启).
输出: results/tables/problem3_init_stop_ablation.csv
"""
import sys, os, time, csv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sparse_reconstruction import make_instance, tau_gpsr, SmoothedL1Quad
from cg_solvers import strong_wolfe_alpha, beta_rule, gpsr_bb_solve

N, K, KTRUE, SIGMA2, SEEDS = 4096, 1024, 160, 1e-4, list(range(10))
MU_SEQ, TOLG = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6), 1e-6
REL_TOL = 1e-6
TOLP_GPSR, MAXIT_GPSR = 1e-3, 3000
MAXIT_SEG, MAXIT_TOTAL = 2000, 12000
TAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "tables")


def l1_obj(A, b, tau, x):
    return float(tau * np.sum(np.abs(x)) + 0.5 * np.sum((A @ x - b) ** 2))


def cg_until_target(A, b, tau, model, x0, target, stop):
    """μ 续延 CG; stop="grad" 每段用梯度准则, stop="rel" 用相对目标变化准则.
    返回达到 1.001×target 的累计迭代/时间 (未达到则 conv=False)."""
    xk = np.asarray(x0, dtype=np.float64).copy()
    acc_it = 0
    t_start = time.perf_counter()
    for mu in MU_SEQ:
        model.set_mu(float(mu))
        gk = model.gradient(xk)
        d = -gk.copy()
        f_cur = model.value(xk)
        for _ in range(MAXIT_SEG):
            if stop == "grad" and float(np.max(np.abs(gk))) <= TOLG:
                break
            alpha, _ = strong_wolfe_alpha(model, xk, d, gk)
            if alpha <= 0.0:
                break
            xk = xk + alpha * d
            g_new = model.gradient(xk)
            f_new = model.value(xk)
            acc_it += 1
            if stop == "rel" and acc_it > 0 and \
                    abs(f_new - f_cur) / max(abs(f_cur), 1e-12) <= REL_TOL:
                gk, f_cur = g_new, f_new
                break
            if l1_obj(A, b, tau, xk) <= 1.001 * target:
                return acc_it, time.perf_counter() - t_start, True
            bb = beta_rule("prp+", g_new, gk, d, g_new - gk)
            d = -g_new + bb * d
            if float(np.dot(g_new, d)) >= -1e-7 * float(np.dot(g_new, g_new)):
                d = -g_new.copy()
            gk, f_cur = g_new, f_new
            if acc_it >= MAXIT_TOTAL:
                return acc_it, time.perf_counter() - t_start, False
    return acc_it, time.perf_counter() - t_start, \
        bool(l1_obj(A, b, tau, xk) <= 1.001 * target)


def main():
    rows = []
    for seed in SEEDS:
        A, x, b = make_instance(m=K, n=N, k_true=KTRUE, sigma2=SIGMA2, seed=seed)
        tau = tau_gpsr(A, b)
        model = SmoothedL1Quad(A, b, tau, mu=1e-3)
        g = gpsr_bb_solve(A, b, tau, tolP=TOLP_GPSR, maxit=MAXIT_GPSR)
        target = l1_obj(A, b, tau, g["x"])
        for name, x0, stop in (("zero_init_grad_stop", np.zeros(N), "grad"),
                               ("atb_init_rel_stop", A.T @ b, "rel")):
            it, t, conv = cg_until_target(A, b, tau, model, x0, target, stop)
            rows.append(dict(seed=seed, config=name, it_to_target=it, t=t,
                             reached_target=conv))
            print(f"seed {seed} {name}: it={it} t={t:.2f}s conv={conv}", flush=True)
    with open(os.path.join(TAB, "problem3_init_stop_ablation.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for name in ("zero_init_grad_stop", "atb_init_rel_stop"):
        sub = [r for r in rows if r["config"] == name and r["reached_target"]]
        if sub:
            print(f"{name}: mean it = {np.mean([r['it_to_target'] for r in sub]):.1f}"
                  f" over {len(sub)}/{len(SEEDS)} reached")
    print("DONE")


if __name__ == "__main__":
    main()
