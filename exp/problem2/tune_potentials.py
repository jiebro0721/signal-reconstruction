"""问题二: 5 种保边势函数的 (α, β) 调参 + 逐势函数 α 敏感性曲线.

调参协议 (v2, 依据 zcode 审查):
  - 先固定 beta=40 扫 α (每种 φ 用其自身尺度网格), 取 PSNR 最优 α*;
  - 再在 α* 下扫 β ∈ {5,20,40,80,160,320,640} (覆盖 logcosh/log1 在 160 处的上升区间);
  - 选优规则: 优先在"收敛"运行中取 PSNR 最大者; 全部未收敛时取最大 PSNR 并标注 truncated;
  - power (|t|^α, α<2) 在最优参数处补一次 maxit=5000 的验证点 (PSNR 是否已到平台);
  - 结果写入 results/tables/problem2_params.json (并由 run_problem2.py 直接读取)。
调参图像: Lena 512, 种子 2026; 初值统一为 AMF 输出 y_amf[cand]。
"""
import sys, os, time, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from noise_model import add_salt_and_pepper, load_gray
from amf import adaptive_median_filter
from restoration_model import Phase2Model
from solvers import gpsr_bb
from metrics import psnr

IMG = os.path.join(ROOT, "data", "test_images", "lena_gray_512.tif")
RES = os.path.join(ROOT, "exp", "problem2", "results")
TAB = os.path.join(RES, "tables")
os.makedirs(TAB, exist_ok=True)

ALPHA_GRID = {
    "sqrt":     [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0],
    "power":    [1.05, 1.1, 1.2, 1.3, 1.4, 1.6, 2.0],
    "logcosh":  [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    "log1":     [1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
    "huber":    [1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
}
BETA_GRID = [5.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0]
MAXIT = 1500


def solve(y, cand, pot_, alpha, beta, x, mu=1.0, maxit=MAXIT):
    m = Phase2Model(y, cand, beta=beta, alpha=alpha, smooth="sqrt", potential=pot_)
    t0 = time.perf_counter()
    res = gpsr_bb(m, u0, mu=mu, tolP=1e-2, maxit=maxit)
    xh = y.copy(); xh[cand] = res["u"]
    return dict(it=int(res["it"]), t=time.perf_counter() - t0,
                conv=bool(res["converged"]), psnr=psnr(x, xh),
                gap=float(res["hist_gap"][-1]))


u0 = None  # 由 main 设置 (AMF 输出)


def pick_best(cands):
    """选优: 优先收敛解中 PSNR 最大; 全部未收敛时取 PSNR 最大 (标注 truncated)。"""
    convs = [c for c in cands if c["conv"]]
    pool = convs if convs else cands
    best = max(pool, key=lambda c: c["psnr"])
    return best, bool(convs)


def main():
    global u0
    only_r = None
    if len(sys.argv) > 1:
        only_r = float(sys.argv[1])        # 可选: 只调一个噪声等级 (并行加速)
    lines = ["# 问题二调参结果 (Lena 512, v2: β 网格扩展至 640, 选优优先收敛解)"]
    params = {}
    x = load_gray(IMG)
    levels = [only_r] if only_r else (0.3, 0.5)
    for r in levels:
        rng = np.random.default_rng(2026)
        y, _ = add_salt_and_pepper(x, p=r / 2, q=r / 2, rng=rng)
        cand, y_amf, _ = adaptive_median_filter(y, r=r)
        u0 = y_amf[cand]
        lines.append(f"\n== r={r:.0%} #cand={cand.sum()} ==")
        for pot_ in ALPHA_GRID:
            # 1) α 扫描 (beta=40)
            cands_a = []
            for a in ALPHA_GRID[pot_]:
                res = solve(y, cand, pot_, a, 40.0, x)
                tag = "trunc" if not res["conv"] else "conv "
                lines.append(f"  [{pot_:8s}] alpha={a:8g} beta=40  it={res['it']:5d} "
                             f"PSNR={res['psnr']:7.3f} {tag} gap={res['gap']:.2e}")
                cands_a.append(dict(alpha=a, **res))
            a_star, _ = pick_best(cands_a)
            a_star = a_star["alpha"]
            # power 特例: α=2 (L2 极限) 仅作为敏感性曲线参考, 不入 α* 选优
            # (1<α<2 才体现该族的"保边"本质; α=2 时该势函数退化为二次惩罚)
            if pot_ == "power" and a_star >= 2.0:
                sub = [c for c in cands_a if c["alpha"] < 2.0]
                a_star = max(sub, key=lambda c: c["psnr"])["alpha"]
                lines.append(f"  >> power: α=2 为 L2 边界参考, 选优限制在 1<α<2 → α*={a_star:g}")
            # 2) β 扫描 (alpha=a*)
            cands_b = []
            for b in BETA_GRID:
                res = solve(y, cand, pot_, a_star, b, x)
                tag = "trunc" if not res["conv"] else "conv "
                lines.append(f"  [{pot_:8s}] alpha={a_star:8g} beta={b:5.0f} it={res['it']:5d} "
                             f"PSNR={res['psnr']:7.3f} {tag} gap={res['gap']:.2e}")
                cands_b.append(dict(beta=b, **res))
            best_b, any_conv = pick_best(cands_b)
            b_star = best_b["beta"]
            params.setdefault(pot_, {})["0.3" if abs(r - 0.3) < 1e-9 else "0.5"] = [
                float(a_star), float(b_star)]
            note = "" if any_conv else "  [注: 全部未收敛 → truncated 最优]"
            lines.append(f"  >> {pot_}: alpha*={a_star:g}, beta*={b_star:g}, "
                         f"PSNR*={best_b['psnr']:.3f}, conv={any_conv}{note}")
            # 3) power 补 maxit=5000 验证 (判断是否已达平台)
            if pot_ == "power":
                res5 = solve(y, cand, pot_, a_star, b_star, x, maxit=5000)
                lines.append(f"  >> power {MAXIT}->5000 步验证: PSNR={res5['psnr']:.4f} "
                             f"gap={res5['gap']:.3e} conv={res5['conv']}")
            print(lines[-1], flush=True)
    out = "\n".join(lines)
    suffix = "" if only_r is None else f"_r{int(round(only_r * 100))}"
    with open(os.path.join(TAB, f"problem2_tuning{suffix}.txt"), "w", encoding="utf-8") as f:
        f.write(out)
    if only_r is None:
        with open(os.path.join(TAB, "problem2_params.json"), "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        print("WROTE problem2_params.json")
    else:
        # 并行模式: 输出本等级参数行, 由 merge_tuning.py 合并
        for pot_, pr in params.items():
            print(f"PARAMS {pot_} {only_r} {pr['0.3' if abs(only_r - 0.3) < 1e-9 else '0.5'][0]} "
                  f"{pr['0.3' if abs(only_r - 0.3) < 1e-9 else '0.5'][1]}")


if __name__ == "__main__":
    main()
