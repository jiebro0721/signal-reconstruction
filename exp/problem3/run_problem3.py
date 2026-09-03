"""问题三主实验: 修正 PRP+ 共轭梯度法 (光滑化+μ续延) vs 文献[1] GPSR-BB.

实例: 10 个随机实例 (seed 0..9), n=4096, k=1024, 160 个 ±1 尖峰, σ²=1e-4,
      A: 高斯行正交化, τ=0.1‖Aᵀb‖∞ (文献[1]式(22)).
方法: CG (prp+/fr/hs+/dy, 强 Wolfe); CG-PRP+ 带 μ 续延; GPSR-BB (闭式 λ) ± 去偏置.
对比口径: ①各自收敛: IT/时间/误差/支撑; ②同目标值(文献[1]方法学):
          达到 GPSR-BB 目标值 ×1.001 所需迭代数/时间.
输出: results/tables/problem3_full.csv, problem3_summary.csv
"""
import sys, os, time, csv, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sparse_reconstruction import make_instance, tau_gpsr, SmoothedL1Quad
from cg_solvers import (nonlinear_cg, cg_with_mu_continuation, gpsr_bb_solve,
                        beta_rule, strong_wolfe_alpha, debias)

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TAB = os.path.join(RES, "tables")
FIG = os.path.join(RES, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

N, K, KTRUE, SIGMA2, SEEDS = 4096, 1024, 160, 1e-4, list(range(10))
MU_SEQ, TOLG = (1e-1, 1e-2, 1e-3, 1e-4), 1e-6
TOLP_GPSR, MAXIT_GPSR = 1e-3, 3000
MAXIT_CG = 4000


def l1_obj(A, b, tau, x):
    return float(tau * np.sum(np.abs(x)) + 0.5 * np.sum((A @ x - b) ** 2))


def support_stats(x, xtrue, thresh=0.05):
    mask = np.abs(x) > thresh
    tp = int(np.sum(mask & (xtrue != 0)))
    fp = int(np.sum(mask & (xtrue == 0)))
    return tp, fp


def cg_until_target(A, b, tau, model, x0, target, mu_seq=MU_SEQ,
                    maxit=MAXIT_CG, tolG=TOLG):
    """μ 续延 CG, 同时跟踪 ℓ1 目标值, 报告达到 1.001×target 的累计迭代/时间."""
    xk = np.asarray(x0, dtype=np.float64).copy()
    acc_it = 0
    t_start = time.perf_counter()
    for mu in mu_seq:
        model.set_mu(float(mu))
        gk = model.gradient(xk)
        d = -gk.copy()
        for _ in range(maxit):
            if float(np.max(np.abs(gk))) <= tolG:
                break
            alpha, _ = strong_wolfe_alpha(model, xk, d, gk)
            if alpha <= 0.0:
                break
            xk = xk + alpha * d
            g_new = model.gradient(xk)
            y_ = g_new - gk
            bb = beta_rule("prp+", g_new, gk, d, y_)
            d = -g_new + bb * d
            if float(np.dot(g_new, d)) >= -1e-7 * float(np.dot(g_new, g_new)):
                d = -g_new.copy()
            gk = g_new
            acc_it += 1
            if l1_obj(A, b, tau, xk) <= 1.001 * target:
                return xk, acc_it, time.perf_counter() - t_start
    return xk, acc_it, time.perf_counter() - t_start


def run_instance(seed):
    A, x, b = make_instance(m=K, n=N, k_true=KTRUE, sigma2=SIGMA2, seed=seed)
    tau = tau_gpsr(A, b)
    model = SmoothedL1Quad(A, b, tau, mu=1e-3)
    x0 = np.zeros(N)
    xnorm = float(np.linalg.norm(x))
    rows = []

    # ---------- GPSR-BB ----------
    t0 = time.perf_counter()
    g = gpsr_bb_solve(A, b, tau, tolP=TOLP_GPSR, maxit=MAXIT_GPSR)
    g_elapsed = time.perf_counter() - t0
    obj_t = l1_obj(A, b, tau, g["x"])
    rows.append(dict(method="GPSR-BB", it=g["it"], t=g_elapsed,
                     rel=np.linalg.norm(g["x"] - x) / xnorm, obj=obj_t,
                     tp=support_stats(g["x"], x)[0], fp=support_stats(g["x"], x)[1],
                     conv=bool(g["conv"]),
                     it_to_target=g["it"], t_to_target=g_elapsed))
    t0 = time.perf_counter()
    gd = gpsr_bb_solve(A, b, tau, tolP=TOLP_GPSR, maxit=MAXIT_GPSR, do_debias=True)
    gd_elapsed = time.perf_counter() - t0
    rows.append(dict(method="GPSR-BB+debias", it=g["it"] + 1, t=gd_elapsed,
                     rel=np.linalg.norm(gd["x"] - x) / xnorm,
                     obj=l1_obj(A, b, tau, gd["x"]),
                     tp=support_stats(gd["x"], x)[0], fp=support_stats(gd["x"], x)[1],
                     conv=True, it_to_target=g["it"], t_to_target=gd_elapsed))

    # ---------- CG: 共轭参数对照 (单段 μ=1e-3, 到 tolG) ----------
    for beta in ("prp+", "fr", "hs+", "dy"):
        model.set_mu(1e-3)
        t0 = time.perf_counter()
        r = nonlinear_cg(model, x0, beta=beta, maxit=MAXIT_CG, tolG=TOLG)
        el = time.perf_counter() - t0
        rows.append(dict(method=f"CG-{beta}", it=r["it"], t=el,
                         rel=np.linalg.norm(r["x"] - x) / xnorm,
                         obj=l1_obj(A, b, tau, r["x"]),
                         tp=support_stats(r["x"], x)[0], fp=support_stats(r["x"], x)[1],
                         conv=bool(r["conv"]),
                         it_to_target=np.nan, t_to_target=np.nan))

    # ---------- CG-PRP+ μ 续延 (主算法) + 同目标值对比 ----------
    t0 = time.perf_counter()
    r = cg_with_mu_continuation(model, x0, mu_seq=MU_SEQ, beta="prp+",
                                maxit=MAXIT_CG, tolG=TOLG)
    cg_elapsed = time.perf_counter() - t0
    xhat = r["x"]
    _, it_at, t_at = cg_until_target(A, b, tau, model, x0, obj_t)
    rows.append(dict(method="CG-PRP+续延", it=r["it_sum"], t=cg_elapsed,
                     rel=np.linalg.norm(xhat - x) / xnorm,
                     obj=l1_obj(A, b, tau, xhat),
                     tp=support_stats(xhat, x)[0], fp=support_stats(xhat, x)[1],
                     conv=bool(r["conv"]),
                     it_to_target=it_at, t_to_target=t_at))
    xd = debias(A, b, xhat)
    rows.append(dict(method="CG-PRP+续延+debias", it=r["it_sum"], t=cg_elapsed,
                     rel=np.linalg.norm(xd - x) / xnorm,
                     obj=l1_obj(A, b, tau, xd),
                     tp=support_stats(xd, x)[0], fp=support_stats(xd, x)[1],
                     conv=bool(r["conv"]),
                     it_to_target=it_at, t_to_target=t_at))
    return rows


def main():
    all_rows = []
    for seed in SEEDS:
        rows = run_instance(seed)
        all_rows += rows
        print(f"seed {seed}: " + " | ".join(
            f"{r['method']}:it={r['it']},rel={r['rel']:.4f}" for r in rows), flush=True)
    with open(os.path.join(TAB, "problem3_full.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    methods = list(dict.fromkeys(r["method"] for r in all_rows))
    with open(os.path.join(TAB, "problem3_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "it_mean", "it_std", "t_mean", "rel_mean", "obj_mean",
                    "tp_mean", "fp_mean", "it_to_target_mean", "t_to_target_mean",
                    "conv_all"])
        for m in methods:
            sub = [q for q in all_rows if q["method"] == m]
            def stat(key, fn=np.mean):
                vals = [q[key] for q in sub
                        if not (isinstance(q[key], float) and np.isnan(q[key]))]
                return float(fn(vals)) if vals else float("nan")
            w.writerow([m, stat("it"), float(np.std([q["it"] for q in sub])),
                        stat("t"), stat("rel"), stat("obj"), stat("tp"), stat("fp"),
                        stat("it_to_target"), stat("t_to_target"),
                        bool(all(q["conv"] for q in sub))])
    with open(os.path.join(TAB, "problem3_config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(n=N, k=K, k_true=KTRUE, sigma2=SIGMA2, mu_seq=MU_SEQ,
                       tolG=TOLG, tolP=TOLP_GPSR, seeds=SEEDS), f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
