"""问题二 β 网格上界验证: 在标定 α* 下补 β=1280 验证点.

依据审阅意见 M4: 主标定 β 网格上界为 640, 9/10 组 β* 落在该上界,
其中 logcosh (30%) 在 β=640 处 PSNR 仍在上升 (320→640: +0.066 dB).
本脚本在各势函数标定 α* 处追加 β=1280 运行, 检验:
  1) β=1280 相对 β=640 的 PSNR 增益是否可忽略 (平台结论成立);
  2) 若仍有上升, 说明 β* 被网格上界低估, 论文按"网格内最优/平台代表值"表述.
协议与 tune_potentials.py 完全一致 (Lena 512, 种子 2026, 初值 AMF 输出,
tolP=1e-2, maxit=1500, 优先收敛解口径). 参照值 β=640 取自 problem2_tuning.txt.
输出: results/tables/problem2_beta1280_verification.txt / .csv
"""
import sys, os, time, csv
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

# 标定 α* (problem2_params.json) 与 β=640 参照 PSNR (problem2_tuning.txt)
ALPHA_STAR = {("sqrt", 0.3): 300.0, ("sqrt", 0.5): 1000.0,
              ("power", 0.3): 1.4, ("power", 0.5): 1.6,
              ("logcosh", 0.3): 0.1, ("logcosh", 0.5): 0.1,
              ("log1", 0.3): 3.0, ("log1", 0.5): 3.0,
              ("huber", 0.3): 30.0, ("huber", 0.5): 30.0}
PSNR_B640 = {("sqrt", 0.3): 38.365, ("sqrt", 0.5): 34.744,
             ("power", 0.3): 38.345, ("power", 0.5): 34.760,
             ("logcosh", 0.3): 38.182, ("logcosh", 0.5): 34.589,
             ("log1", 0.3): 38.313, ("log1", 0.5): 34.615,
             ("huber", 0.3): 38.190, ("huber", 0.5): 34.664}
BETA_NEW = 1280.0
MAXIT = 1500


def solve(y, cand, pot_, alpha, beta, x):
    m = Phase2Model(y, cand, beta=beta, alpha=alpha, smooth="sqrt", potential=pot_)
    t0 = time.perf_counter()
    res = gpsr_bb(m, u0, mu=1.0, tolP=1e-2, maxit=MAXIT)
    xh = y.copy(); xh[cand] = res["u"]
    return dict(it=int(res["it"]), t=time.perf_counter() - t0,
                conv=bool(res["converged"]), psnr=psnr(x, xh),
                gap=float(res["hist_gap"][-1]))


def main():
    x = load_gray(IMG)
    rows, lines = [], ["# 问题二 β=1280 网格上界验证 (Lena 512, 种子 2026, "
                       "α* 取 problem2_params.json, 参照 β=640 取 problem2_tuning.txt)"]
    for r in (0.3, 0.5):
        rng = np.random.default_rng(2026)
        y, _ = add_salt_and_pepper(x, p=r / 2, q=r / 2, rng=rng)
        cand, y_amf, _ = adaptive_median_filter(y, r=r)
        globals()["u0"] = y_amf[cand]
        lines.append(f"\n== r={r:.0%} #cand={cand.sum()} ==")
        for pot_ in ("sqrt", "power", "logcosh", "log1", "huber"):
            a = ALPHA_STAR[(pot_, r)]
            res = solve(y, cand, pot_, a, BETA_NEW, x)
            ref = PSNR_B640[(pot_, r)]
            gain = res["psnr"] - ref
            tag = "conv " if res["conv"] else "trunc"
            lines.append(
                f"  [{pot_:8s}] alpha*={a:8g} beta={BETA_NEW:.0f} it={res['it']:5d} "
                f"PSNR={res['psnr']:7.3f} {tag} gap={res['gap']:.2e} "
                f"| β=640 参照 {ref:.3f} → 增益 {gain:+.3f} dB")
            rows.append(dict(potential=pot_, r=r, alpha=a, beta=BETA_NEW,
                             it=res["it"], psnr=round(res["psnr"], 4),
                             conv=res["conv"], gap=res["gap"],
                             psnr_b640_ref=ref, gain=round(gain, 4)))
            print(lines[-1], flush=True)
    with open(os.path.join(TAB, "problem2_beta1280_verification.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(TAB, "problem2_beta1280_verification.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("DONE")


if __name__ == "__main__":
    main()
