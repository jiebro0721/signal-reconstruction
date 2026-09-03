"""问题一主实验: 两阶段椒盐噪声恢复 (AMF 检测 + 光滑化 GPSR-BB 求解).

图像: 12 张 512×512 灰度标准测试图; 噪声等级 r = 30%, 50%; 每配置 3 个随机种子.
输出: results/tables/*.csv, results/figures/*.png, results/restored/*.tif

主配置: 模型(2)(含 L1 数据项), φα=√(t²+α), α=300, β=40; 光滑参数 μ=1;
        GPSR-BB (BB1 + 投影 + 非单调 Armijo), tolP=1e-2, maxit=1500.
"""
import sys, os, time, json, csv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from noise_model import add_salt_and_pepper, load_gray
from amf import adaptive_median_filter, w_max_for_noise_level
from restoration_model import Phase2Model
from solvers import gpsr_bb
from metrics import psnr, snr, mae, detection_stats

IMG_DIR = os.path.join(ROOT, "data", "test_images")
RES = os.path.join(ROOT, "exp", "problem1", "results")
TAB = os.path.join(RES, "tables")
FIG = os.path.join(RES, "figures")
RST = os.path.join(RES, "restored")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
os.makedirs(RST, exist_ok=True)

GRAY_IMAGES = ["cameraman.tif", "house.tif", "jetplane.tif", "lake.tif",
               "lena_gray_512.tif", "livingroom.tif", "mandril_gray.tif",
               "peppers_gray.tif", "pirate.tif", "walkbridge.tif",
               "woman_blonde.tif", "woman_darkhair.tif"]
SEEDS = [2026, 7, 42]
BETA, ALPHA, MU = 40.0, 300.0, 1.0
TOLP, MAXIT = 1e-2, 1500


def run_case(image, r, seed, save=True):
    rng = np.random.default_rng(seed)
    x = load_gray(os.path.join(IMG_DIR, image))
    y, true_mask = add_salt_and_pepper(x, p=r / 2, q=r / 2, rng=rng)
    wmax = w_max_for_noise_level(r)
    t0 = time.perf_counter()
    cand, y_amf, _ = adaptive_median_filter(y, r=r)
    t_amf = time.perf_counter() - t0
    model = Phase2Model(y, cand, beta=BETA, alpha=ALPHA, smooth="sqrt")
    t0 = time.perf_counter()
    res = gpsr_bb(model, y_amf[cand], mu=MU, tolP=TOLP, maxit=MAXIT)
    t_solve = time.perf_counter() - t0
    xh = y.copy()
    xh[cand] = res["u"]
    # 基线对照(文献[2] IV: 中值滤波与 AMF 直接输出)
    from scipy import ndimage
    med3 = ndimage.median_filter(y, size=3, mode="reflect")
    med7 = ndimage.median_filter(y, size=7, mode="reflect")
    det = detection_stats(true_mask, cand)
    row = dict(image=image, r=r, seed=seed, wmax=wmax,
               n_cand=det["n_cand"], tpr=det["tpr"], fpr=det["fpr"],
               it=int(res["it"]), t_amf=t_amf, t_solve=t_solve,
               psnr=psnr(x, xh), snr=snr(x, xh), mae=mae(x, xh),
               psnr_n=psnr(x[cand], xh[cand]), converged=bool(res["converged"]),
               gap=float(res["hist_gap"][-1]),
               psnr_noisy=psnr(x, y), psnr_med3=psnr(x, med3),
               psnr_med7=psnr(x, med7), psnr_amf=psnr(x, y_amf),
               snr_amf=snr(x, y_amf))
    if save:
        import tifffile
        tifffile.imwrite(os.path.join(RST, f"{os.path.splitext(image)[0]}_r{int(r*100)}_s{seed}.tif"),
                         np.clip(np.round(xh), 0, 255).astype(np.uint8))
        np.save(os.path.join(RST, f"{os.path.splitext(image)[0]}_r{int(r*100)}_s{seed}_mask.npy"),
                cand)
    return row


def main():
    rows = []
    for image in GRAY_IMAGES:
        for r in (0.3, 0.5):
            for seed in SEEDS:
                row = run_case(image, r, seed)
                rows.append(row)
                print(f"{image:22s} r={r:.0%} s={seed:5d}: it={row['it']:5d} "
                      f"t={row['t_solve']:6.2f}s PSNR={row['psnr']:7.3f} "
                      f"SNR={row['snr']:7.3f}", flush=True)
    # 原始明细表
    with open(os.path.join(TAB, "problem1_full.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # 均值汇总(按 图像×噪声等级)
    summary = {}
    for image in GRAY_IMAGES:
        for r in (0.3, 0.5):
            sub = [q for q in rows if q["image"] == image and abs(q["r"] - r) < 1e-9]
            summary[(image, r)] = sub
    keys = ["it", "t_amf", "t_solve", "psnr", "snr", "mae", "psnr_n", "tpr", "fpr",
            "psnr_noisy", "psnr_med3", "psnr_med7", "psnr_amf", "snr_amf"]
    with open(os.path.join(TAB, "problem1_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "r", "n_cand", "wmax"] + [f"{k}_mean" for k in keys] +
                   [f"{k}_std" for k in keys])
        for (image, r), sub in summary.items():
            vals = {k: [q[k] for q in sub] for k in keys}
            w.writerow([image, r, sub[0]["n_cand"], sub[0]["wmax"]] +
                       [float(np.mean(vals[k])) for k in keys] +
                       [float(np.std(vals[k])) for k in keys])
    with open(os.path.join(TAB, "problem1_config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(beta=BETA, alpha=ALPHA, mu=MU, tolP=TOLP, maxit=MAXIT,
                       seeds=SEEDS, images=GRAY_IMAGES), f, indent=2, ensure_ascii=False)
    print("DONE")


if __name__ == "__main__":
    main()
