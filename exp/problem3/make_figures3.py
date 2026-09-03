"""问题三图: (a) 收敛曲线对比; (b) 信号恢复对比; (c) 汇总条形图."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sparse_reconstruction import make_instance, tau_gpsr, SmoothedL1Quad
from cg_solvers import (gpsr_bb_solve, beta_rule, strong_wolfe_alpha,
                        cg_with_mu_continuation, debias)

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TAB = os.path.join(RES, "tables")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

_zh = [f.name for f in font_manager.fontManager.ttflist
       if any(k in f.name for k in ("YaHei", "SimHei", "SimSun", "Noto Sans CJK"))]
plt.rcParams["font.sans-serif"] = _zh + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9


def l1_obj(A, b, tau, x):
    return float(tau * np.sum(np.abs(x)) + 0.5 * np.sum((A @ x - b) ** 2))


def cg_hist(A, b, tau, model, maxit=600, mu_seq=None):
    """记录 PRP+ CG 的 (迭代, ℓ1目标) 历史; mu_seq=None 为单段."""
    xs = np.zeros_like(b) if False else None
    xs = np.zeros(model.A.shape[1])
    acc = 0
    h = [(0, l1_obj(A, b, tau, xs), 0.0)]
    if mu_seq is None:
        mu_seq = (model.mu,)
    for mu in mu_seq:
        model.set_mu(mu)
        gk = model.gradient(xs)
        d = -gk.copy()
        while acc < maxit:
            if float(np.max(np.abs(gk))) <= 1e-6:
                break
            alpha, _ = strong_wolfe_alpha(model, xs, d, gk)
            if alpha <= 0.0:
                break
            xs = xs + alpha * d
            g_new = model.gradient(xs)
            bb = beta_rule("prp+", g_new, gk, d, g_new - gk)
            d = -g_new + bb * d
            if float(np.dot(g_new, d)) >= -1e-7 * float(np.dot(g_new, g_new)):
                d = -g_new.copy()
            gk = g_new
            acc += 1
            h.append((acc, l1_obj(A, b, tau, xs), float(acc)))
    return h, xs


def fig_convergence(seed=0):
    A, x, b = make_instance(seed=seed)
    tau = tau_gpsr(A, b)
    model = SmoothedL1Quad(A, b, tau, mu=1e-3)
    g = gpsr_bb_solve(A, b, tau, tolP=1e-3, maxit=3000)
    hg = np.array(g["hist_f"])           # BCQP 目标(含常数项), 仅看趋势
    hcg, xs = cg_hist(A, b, tau, model, maxit=600)
    hcont, xcont = cg_hist(A, b, tau, model, maxit=400, mu_seq=(1e-1, 1e-2, 1e-3))
    fmin = min(hg.min(), min(h[1] for h in hcont), min(h[1] for h in hcg))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.0))
    ax = axes[0]
    ax.semilogy(np.arange(1, len(hg) + 1), hg - hg[-1] + 1e-12, lw=1.5,
                label="GPSR-BB  (BCQP 目标)", color="#4c72b0")
    ax.semilogy([h[0] for h in hcg], np.array([h[1] for h in hcg]) - fmin + 1e-12,
                lw=1.5, label="CG-PRP+ (单段 μ=1e-3)", color="#dd8452")
    ax.semilogy([h[0] for h in hcont], np.array([h[1] for h in hcont]) - fmin + 1e-12,
                lw=1.5, label="CG-PRP+ (μ 续延)", color="#55a868")
    ax.set_xlabel("iteration k"); ax.set_ylabel("objective gap (log)")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_title(f"seed {seed}: 目标函数收敛对比")
    ax = axes[1]
    ax.stem(np.arange(200), x[:200], linefmt="C0-", markerfmt="C0o", basefmt=" ",
            label="真实 x")
    ax.stem(np.arange(200), g["x"][:200], linefmt="C1-", markerfmt="C1x", basefmt=" ",
            label="GPSR-BB")
    ax.stem(np.arange(200), xcont[:200], linefmt="C2--", markerfmt="C2+", basefmt=" ",
            label="CG-PRP+ 续延")
    ax.set_xlabel("index i"); ax.set_ylabel("x_i")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_title("信号恢复对比 (前 200 分量)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "convergence_signal.png"), dpi=200)
    plt.close(fig)
    print("fig: convergence_signal")


def fig_bars():
    import csv
    rows = list(csv.DictReader(open(os.path.join(TAB, "problem3_summary.csv"),
                                    encoding="utf-8-sig")))
    cm = {q["method"]: q for q in rows}
    order = ["GPSR-BB", "GPSR-BB+debias", "CG-prp+", "CG-fr", "CG-hs+", "CG-dy",
             "CG-PRP+续延", "CG-PRP+续延+debias"]
    labels = ["GPSR-BB", "GPSR-BB\n+debias", "CG-PRP+\nμ=1e-3", "CG-FR", "CG-HS+",
              "CG-DY", "CG-PRP+\nμ续延", "CG-PRP+续延\n+debias"]
    sel = [(lab, m) for lab, m in zip(labels, order) if m in cm]
    rel = [float(cm[m]["rel_mean"]) for _, m in sel]
    it = [float(cm[m]["it_mean"]) for _, m in sel]
    tme = [float(cm[m]["t_mean"]) for _, m in sel]
    lab = [l for l, _ in sel]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0))
    for ax, vals, ylab, tt in zip(axes, [rel, it, tme],
                                  ["relative error $\\|\\hat x-x\\|/\\|x\\|$",
                                   "iterations", "time (s)"],
                                  ["恢复误差", "迭代次数", "计算时间"]):
        ax.bar(np.arange(len(vals)), vals, color="#4c72b0")
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(lab, fontsize=7)
        ax.set_ylabel(ylab); ax.set_title(tt)
        ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "summary_bars.png"), dpi=200)
    plt.close(fig)
    print("fig: summary_bars")


def main():
    fig_convergence(seed=0)
    fig_bars()
    print("done")


if __name__ == "__main__":
    main()
