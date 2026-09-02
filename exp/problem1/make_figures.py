"""图: 收敛曲线 + 视觉对比 + PSNR 汇总条形图."""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from noise_model import add_salt_and_pepper, load_gray
from amf import adaptive_median_filter
from restoration_model import Phase2Model
from solvers import gpsr_bb
from metrics import psnr, snr

D = os.path.join(ROOT, "data", "test_images")
FIG = os.path.join(ROOT, "exp", "problem1", "results", "figures")
BETA, ALPHA, MU = 40.0, 300.0, 1.0
_zh = [f.name for f in font_manager.fontManager.ttflist
       if any(k in f.name for k in ("YaHei", "SimHei", "SimSun", "Noto Sans CJK", "Source Han"))]
plt.rcParams["font.sans-serif"] = _zh + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9


def fig_convergence(x, y, cand, y_amf, r, img, stem=None):
    m = Phase2Model(y, cand, beta=BETA, alpha=ALPHA, smooth="sqrt")
    res = gpsr_bb(m, y_amf[cand], mu=MU, tolP=1e-2, maxit=1500)
    xh = y.copy(); xh[cand] = res["u"]
    it = np.arange(1, len(res["hist_gap"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6))
    axes[0].semilogy(it, res["hist_gap"], lw=1.4)
    axes[0].set_xlabel("iteration $k$"); axes[0].set_ylabel("projection gap")
    axes[0].set_title(f"{img} $r$={r:.0%}, GPSR-BB, IT={res['it']}")
    axes[0].grid(alpha=.3)
    f0 = res["hist_f"][0]
    axes[1].semilogy(it, np.abs((res["hist_f"] - res["hist_f"][-1]) / f0) + 1e-16, lw=1.4)
    axes[1].set_xlabel("iteration $k$"); axes[1].set_ylabel("relative obj. error")
    axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"conv_{stem}_r{int(r*100)}.png"), dpi=200)
    plt.close(fig)
    return res


def fig_visual(img, r, x, y, y_amf, xh, det=None):
    fig, ax = plt.subplots(2, 4, figsize=(9.4, 5.2))
    panels = [(x, "原图"), (y, f"噪声图 r={r:.0%}"), (y_amf, "AMF 输出"), (xh, "GPSR 恢复")]
    for j, (a, t) in enumerate(panels):
        ax[0, j].imshow(a, cmap="gray", vmin=0, vmax=255)
        ax[0, j].set_title(t, fontsize=9)
        ax[0, j].axis("off")
    # 局部放大 300,100:-: 取块
    d = 96
    for j, (a, t) in enumerate(panels):
        blk = a[200:200 + d, 200:200 + d]
        ax[1, j].imshow(blk, cmap="gray", vmin=0, vmax=255)
        ax[1, j].set_title("局部放大", fontsize=8)
        ax[1, j].axis("off")
    fig.suptitle(f"{img}: PSNR={psnr(x, xh):.2f} dB, SNR={snr(x, xh):.2f} dB, IT={det['it']}, t={det['t']:.2f}s",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"vis_{img[:-4]}_r{int(r*100)}.png"), dpi=200)
    plt.close(fig)


def main():
    # 1) 收敛曲线: Lena 30%/50%, Cameraman 50%
    for img, r in [("lena_gray_512.tif", 0.3), ("lena_gray_512.tif", 0.5),
                   ("cameraman.tif", 0.5)]:
        x = load_gray(os.path.join(D, img))
        rng = np.random.default_rng(2026)
        y, _ = add_salt_and_pepper(x, p=r / 2, q=r / 2, rng=rng)
        cand, y_amf, _ = adaptive_median_filter(y, r=r)
        stem = os.path.splitext(img)[0]
        res = fig_convergence(x, y, cand, y_amf, r, stem, stem=stem)
        print(f"conv fig {img} r={r:.0%} IT={res['it']} conv={res['converged']}")

    # 2) 视觉对比: Lena/Cameraman/Peppers @30%,50%
    for img in ["lena_gray_512.tif", "cameraman.tif", "peppers_gray.tif"]:
        x = load_gray(os.path.join(D, img))
        for r in (0.3, 0.5):
            rng = np.random.default_rng(2026)
            y, _ = add_salt_and_pepper(x, p=r / 2, q=r / 2, rng=rng)
            cand, y_amf, _ = adaptive_median_filter(y, r=r)
            m = Phase2Model(y, cand, beta=BETA, alpha=ALPHA, smooth="sqrt")
            t0 = time.perf_counter()
            res = gpsr_bb(m, y_amf[cand], mu=MU, tolP=1e-2, maxit=1500)
            xh = y.copy(); xh[cand] = res["u"]
            fig_visual(img, r, x, y, y_amf, xh,
                       dict(it=res["it"], t=time.perf_counter() - t0))
            print(f"vis fig {img} r={r:.0%} IT={res['it']}")

    # 3) PSNR 汇总条形图 (从 summary csv)
    import csv
    rows = list(csv.DictReader(open(
        os.path.join(ROOT, "exp", "problem1", "results", "tables", "problem1_summary.csv"),
        encoding="utf-8-sig")))
    imgs = [q["image"] for q in rows if q["r"] == "0.3"]
    p30 = [float(q["psnr_mean"]) for q in rows if q["r"] == "0.3"]
    p50 = [float(q["psnr_mean"]) for q in rows if q["r"] == "0.5"]
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    xpos = np.arange(len(imgs)); wdt = 0.38
    ax.bar(xpos - wdt / 2, p30, wdt, label="r=30%", color="#4c72b0")
    ax.bar(xpos + wdt / 2, p50, wdt, label="r=50%", color="#dd8452")
    ax.set_xticks(xpos); ax.set_xticklabels([q[:-4] for q in imgs], rotation=30, ha="right")
    ax.set_ylabel("PSNR (dB)"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "psnr_summary.png"), dpi=200)
    print("done")


if __name__ == "__main__":
    main()
