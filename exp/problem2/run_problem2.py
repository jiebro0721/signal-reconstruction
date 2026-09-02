"""问题二主实验: 5 种保边势函数在各自调优 (α*, β*) 下的恢复效果对比.

图像: Lena / Cameraman / Peppers (512), 噪声 30%/50%, 每配置 2 个种子;
指标: PSNR, SNR, SSIM, 边缘 PSNR, MAE, Ω-PSNR, 迭代次数, 时间;
输出: results/tables/problem2_full.csv, problem2_summary.csv
"""
import sys, os, time, csv, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from noise_model import add_salt_and_pepper, load_gray
from amf import adaptive_median_filter
from restoration_model import Phase2Model
from solvers import gpsr_bb
from metrics import psnr, snr, mae, ssim, edge_psnr, edge_mask

IMG_DIR = os.path.join(ROOT, "data", "test_images")
RES = os.path.join(ROOT, "exp", "problem2", "results")
TAB = os.path.join(RES, "tables")
FIG = os.path.join(RES, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

IMGS = ["lena_gray_512.tif", "cameraman.tif", "peppers_gray.tif"]
SEEDS = [2026, 7]
RLEVELS = (0.3, 0.5)
MU, TOLP, MAXIT = 1.0, 1e-2, 1500

# 调优结果 (由 tune_potentials.py 得到, 运行时若存在 config 则读取)
PARAMS_DEFAULT = {   # (alpha*, beta*) —— 以 Lena 30%/50% 调优结果为准
    "sqrt":    (300.0, 40.0),
    "power":   (1.3, 40.0),
    "logcosh": (0.03, 40.0),
    "log1":    (30.0, 40.0),
    "huber":   (30.0, 40.0),
}
POT_NAMES = {"sqrt": r"$\sqrt{t^2+\alpha}$", "power": r"$|t|^{\alpha}$",
             "logcosh": r"$\log\cosh(\alpha t)$", "log1": r"$|t|/\alpha-\log(1+|t|/\alpha)$",
             "huber": r"Huber"}


def load_params():
    """返回 {potential: {r(str): [alpha, beta]}} 或 {potential: [alpha, beta]}."""
    p = os.path.join(TAB, "problem2_params.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return {k: {str(r): v for r in (0.3, 0.5)} for k, v in PARAMS_DEFAULT.items()}


def get_pv(params, pot_, r):
    v = params[pot_]
    if isinstance(v, dict):
        key = "0.3" if abs(r - 0.3) < 1e-9 else "0.5"
        return v[key][0], v[key][1]
    return v[0], v[1]


def run_case(image, r, seed, params, save=False):
    rng = np.random.default_rng(seed)
    x = load_gray(os.path.join(IMG_DIR, image))
    y, _ = add_salt_and_pepper(x, p=r / 2, q=r / 2, rng=rng)
    cand, y_amf, _ = adaptive_median_filter(y, r=r)
    from amf import w_max_for_noise_level
    from metrics import edge_mask
    wmax = w_max_for_noise_level(r)
    em = edge_mask(x)
    edge_frac = float(em.mean())
    rows = []
    for pot_ in params:
        alpha, beta = get_pv(params, pot_, r)
        m = Phase2Model(y, cand, beta=beta, alpha=alpha, smooth="sqrt", potential=pot_)
        t0 = time.perf_counter()
        res = gpsr_bb(m, y_amf[cand], mu=MU, tolP=TOLP, maxit=MAXIT)
        t_solve = time.perf_counter() - t0
        xh = y.copy(); xh[cand] = res["u"]
        rows.append(dict(image=image, r=r, seed=seed, potential=pot_,
                         alpha=alpha, beta=beta, wmax=wmax, edge_frac=edge_frac,
                         it=int(res["it"]), t=t_solve,
                         psnr=psnr(x, xh), snr=snr(x, xh), mae=mae(x, xh),
                         ssim=ssim(x, xh), edge_psnr=edge_psnr(x, xh),
                         psnr_n=psnr(x[cand], xh[cand]), converged=bool(res["converged"])))
    return rows


def main():
    params = load_params()
    rows = []
    for image in IMGS:
        for r in RLEVELS:
            for seed in SEEDS:
                rows += run_case(image, r, seed, params)
                print(f"{image} r={r:.0%} s={seed} done", flush=True)
    with open(os.path.join(TAB, "problem2_full.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    keys = ["psnr", "snr", "ssim", "edge_psnr", "mae", "psnr_n", "it", "t"]
    # 汇总(全部图像) 与 汇总(排除标定集 Lena —— out-of-sample)
    blocks = [("problem2_summary.csv", rows),
              ("problem2_summary_oos.csv", [q for q in rows if q["image"] != "lena_gray_512.tif"])]
    for fname, pool in blocks:
        with open(os.path.join(TAB, fname), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["potential", "r"] + [f"{k}_mean" for k in keys] +
                       [f"{k}_std" for k in keys] + ["alpha", "beta"])
            for pot_ in params:
                for r in RLEVELS:
                    sub = [q for q in pool if q["potential"] == pot_ and abs(q["r"] - r) < 1e-9]
                    if not sub:
                        continue
                    vals = {k: [q[k] for q in sub] for k in keys}
                    a, b = get_pv(params, pot_, r)
                    w.writerow([pot_, r] + [float(np.mean(vals[k])) for k in keys] +
                               [float(np.std(vals[k])) for k in keys] + [a, b])
    print("DONE")


if __name__ == "__main__":
    main()
