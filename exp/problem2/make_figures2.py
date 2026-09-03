"""问题二作图: (1) 各势函数 PSNR-α 曲线; (2) 最终指标条形图; (3) 恢复对比图."""
import sys, os, re, time, json
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
from metrics import psnr

IMG_DIR = os.path.join(ROOT, "data", "test_images")
RES = os.path.join(ROOT, "exp", "problem2", "results")
TAB = os.path.join(RES, "tables")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

_zh = [f.name for f in font_manager.fontManager.ttflist
       if any(k in f.name for k in ("YaHei", "SimHei", "SimSun", "Noto Sans CJK"))]
plt.rcParams["font.sans-serif"] = _zh + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

POT_LABEL = {"sqrt": r"$\varphi_1\!=\!\sqrt{t^2+\alpha}$",
             "power": r"$\varphi_2\!=\!|t|^{\alpha}$",
             "logcosh": r"$\varphi_3\!=\!\log\cosh(\alpha t)$",
             "log1": r"$\varphi_4\!=\!|t|/\alpha\!-\!\log(1\!+\!|t|/\alpha)$",
             "huber": r"$\varphi_5$ (Huber)"}
COLOR = {"sqrt": "#4c72b0", "power": "#dd8452", "logcosh": "#55a868",
         "log1": "#c44e52", "huber": "#8172b3"}


def parse_tuning():
    """解析 tune_potentials.py 的日志, 返回 {(pot, r): [(alpha, psnr), ...]}."""
    txt = open(os.path.join(TAB, "problem2_tuning.txt"), encoding="utf-8").read()
    out = {}
    cur_r = None
    n_matches = 0
    for line in txt.splitlines():
        m = re.match(r"== r=(\d+)%", line)
        if m:
            cur_r = float(m.group(1)) / 100.0
            continue
        m = re.match(r"\s*\[\s*(\w+)\s*\]\s+alpha=\s*([0-9.eE+-]+)\s+beta=40\s+it=\s*(\d+).*PSNR=\s*([0-9.]+)", line)
        if m and cur_r is not None:
            out.setdefault((m.group(1), cur_r), []).append(
                (float(m.group(2)), float(m.group(4))))
            n_matches += 1
    assert n_matches > 0, "调参日志解析行数为 0 (正则或日志格式错误)"
    return out


def fig_alpha_curves(data):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    for r, ax in zip((0.3, 0.5), axes):
        for pot_ in POT_LABEL:
            pts = sorted(data.get((pot_, r), []))
            if not pts:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.plot(xs, ys, "o-", ms=4, lw=1.3, color=COLOR[pot_], label=POT_LABEL[pot_])
        ax.set_xscale("log")
        ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"Lena 512, r={r:.0%} (β=40)")
        ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "psnr_vs_alpha.png"), dpi=200)
    plt.close(fig)
    print("fig: psnr_vs_alpha")


def fig_metrics():
    import csv
    rows = list(csv.DictReader(open(os.path.join(TAB, "problem2_summary.csv"),
                                    encoding="utf-8-sig")))
    pots = ["sqrt", "power", "logcosh", "log1", "huber"]
    keys_def = [(0.3, "psnr_mean", "PSNR (dB)"), (0.3, "ssim_mean", "SSIM"),
                (0.3, "it_mean", "Iterations")]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    for ax, (r, key, ylab) in zip(axes, keys_def):
        sub = [float(q[key]) for q in rows if q["potential"] in pots and abs(float(q["r"]) - r) < 1e-9]
        ax.bar(pots, sub, color=[COLOR[p] for p in pots])
        ax.set_xticklabels([r"$\varphi_1$", r"$\varphi_2$", r"$\varphi_3$", r"$\varphi_4$", r"$\varphi_5$"])
        ax.set_ylabel(ylab); ax.set_title(f"r={r:.0%}, 3 图×2 种子均值")
        ax.grid(alpha=.3, axis="y")
        if key != "it_mean":                 # y 轴截断以示出微小差异
            lo = min(sub); hi = max(sub)
            pad = max(0.06 * (hi - lo), 1e-3)
            ax.set_ylim(lo - pad, hi + pad)
        else:
            ax.set_ylim(0, max(sub) * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "metrics_by_potential.png"), dpi=200)
    plt.close(fig)
    print("fig: metrics_by_potential")


def fig_restored(image="lena_gray_512.tif", r=0.3, params=None):
    import json
    if params is None:
        params = json.load(open(os.path.join(TAB, "problem2_params.json")))
    x = load_gray(os.path.join(IMG_DIR, image))
    rng = np.random.default_rng(2026)
    y, _ = add_salt_and_pepper(x, p=r / 2, q=r / 2, rng=rng)
    cand, y_amf, _ = adaptive_median_filter(y, r=r)
    pots = ["sqrt", "power", "logcosh", "log1", "huber"]
    panels = [("噪声图", y), ("AMF", y_amf)]
    for pot_ in pots:
        v = params[pot_]
        if isinstance(v, dict):
            alpha, beta = v["0.3" if abs(r - 0.3) < 1e-9 else "0.5"]
        else:
            alpha, beta = v
        m = Phase2Model(y, cand, beta=beta, alpha=alpha, smooth="sqrt", potential=pot_)
        res = gpsr_bb(m, y_amf[cand], mu=1.0, tolP=1e-2, maxit=1500)
        xh = y.copy(); xh[cand] = res["u"]
        panels.append((f"{POT_LABEL[pot_]}  PSNR={psnr(x, xh):.2f}dB", xh))
        print(f"  restored {pot_} IT={res['it']}")
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 3.4))
    for ax, (t, a) in zip(axes, panels):
        ax.imshow(a, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t, fontsize=8); ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"restored_{image[:-4]}_r{int(r*100)}.png"), dpi=200)
    plt.close(fig)
    print("fig: restored panels")


def fig_potential_shapes(params=None):
    """五种势函数的形状与导数对比图（机理分析用; 参数取 30% 标定值）."""
    import json
    if params is None:
        params = json.load(open(os.path.join(TAB, "problem2_params.json")))
    t = np.linspace(-150, 150, 600)
    alphas = {}
    for pot_ in POT_LABEL:
        v = params[pot_]
        alphas[pot_] = v["0.3"][0] if isinstance(v, dict) else v[0]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    for pot_ in POT_LABEL:
        from restoration_model import POTENTIALS as PP
        f, df, fs, dfs = PP[pot_]
        a = alphas[pot_]
        axes[0].plot(t, np.where(np.abs(t) < 1e-12, 0, f(t, a)), lw=1.3,
                     color=COLOR[pot_], label=POT_LABEL[pot_])
        axes[1].plot(t, df(t, a), lw=1.3, color=COLOR[pot_])
    axes[0].axhline(255, color="gray", ls=":", lw=0.8)
    axes[0].set_xlabel("$t$"); axes[0].set_ylabel(r"$\varphi_\alpha(t)$")
    axes[0].set_title("势函数形状   $|t|$ 虚线为参考")
    axes[0].grid(alpha=.3); axes[0].legend(fontsize=7)
    axes[1].axhline(1.0, color="gray", ls=":", lw=0.8)
    axes[1].axhline(-1.0, color="gray", ls=":", lw=0.8)
    axes[1].set_xlabel("$t$"); axes[1].set_ylabel(r"$\varphi_\alpha'(t)$")
    axes[1].set_title("势函数导数   ±1 虚线为渐近界")
    axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "potential_shapes.png"), dpi=200)
    plt.close(fig)
    print("fig: potential_shapes")


def fig_beta_sensitivity():
    """从调参日志提取各势函数的 β 敏感性曲线（在各自 α* 处）."""
    txt = open(os.path.join(TAB, "problem2_tuning.txt"), encoding="utf-8").read()
    cur_r = None
    data = {}
    for line in txt.splitlines():
        m = re.match(r"== r=(\d+)%", line)
        if m:
            cur_r = float(m.group(1)) / 100.0
            continue
        m = re.match(r"\s*\[\s*(\w+)\s*\]\s+alpha=\s*([0-9.eE+-]+)"
                     r"\s+beta=\s*([0-9.]+)\s+it=\s*(\d+).*PSNR=\s*([0-9.]+)", line)
        if m and cur_r is not None:
            beta = float(m.group(3))
            if beta != 40.0:   # 只取 β 扫描行
                data.setdefault((m.group(1), cur_r), []).append(
                    (beta, float(m.group(5))))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    for r, ax in zip((0.3, 0.5), axes):
        for pot_ in POT_LABEL:
            pts = sorted(data.get((pot_, r), []))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=4,
                        lw=1.3, color=COLOR[pot_], label=POT_LABEL[pot_])
        ax.set_xscale("log"); ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(r"Lena 512, r=%.0f%% (α=α$^*$)" % (r * 100))
        ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "beta_sensitivity.png"), dpi=200)
    plt.close(fig)
    print("fig: beta_sensitivity")


def main():
    data = parse_tuning()
    fig_alpha_curves(data)
    fig_metrics()
    params = json.load(open(os.path.join(TAB, "problem2_params.json")))
    fig_potential_shapes(params)
    fig_beta_sensitivity()
    fig_restored("lena_gray_512.tif", 0.3, params)
    print("done")


if __name__ == "__main__":
    main()
