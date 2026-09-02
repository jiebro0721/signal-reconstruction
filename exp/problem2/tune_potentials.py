"""问题二: 5 种保边势函数的 (α, β) 调参 + 逐势函数 α 敏感性曲线.

调参协议: 先在固定 beta=40 下扫 α (每种 φ 用其自身尺度网格), 取 PSNR 最优 α*;
再在 α* 下扫 β ∈ {5,20,40,80,160}, 取 PSNR 最优 β*。调参在 Lena 512 (30% 与 50%)
上进行, 结果用于最终对比实验 (run_problem2.py) 的 (α*, β*)。
"""
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
TAB = os.path.join(ROOT, "exp", "problem2", "results", "tables")
os.makedirs(TAB, exist_ok=True)

# 每种势函数的 α 网格 (按其参数语义设定: sqrt 为 t²+α 的 α; power 为指数;
# logcosh 为速率因子; log1 与 huber 为阈值尺度)
ALPHA_GRID = {
    "sqrt":     [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0],
    "power":    [1.05, 1.1, 1.2, 1.3, 1.4, 1.6, 2.0],
    "logcosh":  [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    "log1":     [1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
    "huber":    [1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
}
BETA_GRID = [5.0, 20.0, 40.0, 80.0, 160.0]


def solve(y, cand, pot_, alpha, beta, x, mu=1.0, maxit=1500):
    m = Phase2Model(y, cand, beta=beta, alpha=alpha, smooth="sqrt", potential=pot_)
    t0 = time.perf_counter()
    res = gpsr_bb(m, y[cand], mu=mu, tolP=1e-2, maxit=maxit)
    xh = y.copy(); xh[cand] = res["u"]
    return dict(it=int(res["it"]), t=time.perf_counter() - t0,
                conv=bool(res["converged"]), psnr=psnr(x, xh))


def main():
    lines = ["# 问题二调参结果 (Lena 512)"]
    x = load_gray(IMG)
    for r in (0.3, 0.5):
        rng = np.random.default_rng(2026)
        y, _ = add_salt_and_pepper(x, p=r / 2, q=r / 2, rng=rng)
        cand, y_amf, _ = adaptive_median_filter(y, r=r)
        lines.append(f"\n== r={r:.0%} #cand={cand.sum()} ==")
        for pot_ in ALPHA_GRID:
            # 1) α 扫描 (beta=40)
            best = None
            for a in ALPHA_GRID[pot_]:
                res = solve(y, cand, pot_, a, 40.0, x)
                tag = f"alpha={a:8g} beta=40  it={res['it']:5d} t={res['t']:5.2f} PSNR={res['psnr']:7.3f} conv={res['conv']}"
                lines.append(f"  [{pot_:8s}] {tag}")
                if best is None or res["psnr"] > best[1]:
                    best = (a, res["psnr"])
            a_star = best[0]
            # 2) β 扫描 (alpha=a*)
            bestb = None
            for b in BETA_GRID:
                res = solve(y, cand, pot_, a_star, b, x)
                lines.append(f"  [{pot_:8s}] alpha={a_star:8g} beta={b:5.0f} it={res['it']:5d} t={res['t']:5.2f} PSNR={res['psnr']:7.3f}")
                if bestb is None or res["psnr"] > bestb[1]:
                    bestb = (b, res["psnr"])
            lines.append(f"  >> {pot_}: alpha*={a_star:g}, beta*={bestb[0]:g}, PSNR*={bestb[1]:.3f}")
            print(lines[-1], flush=True)
    out = "\n".join(lines)
    with open(os.path.join(TAB, "problem2_tuning.txt"), "w", encoding="utf-8") as f:
        f.write(out)


if __name__ == "__main__":
    main()
