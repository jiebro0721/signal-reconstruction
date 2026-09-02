"""评价指标: PSNR / SNR / MAE / 噪声检测统计."""
import numpy as np


def psnr(x, xhat, peak=255.0):
    """峰值信噪比 PSNR = 10 log10( peak² / MSE )."""
    mse = float(np.mean((x - xhat) ** 2))
    if mse <= 1e-300:
        return float("inf")
    return 10.0 * np.log10(peak * peak / mse)


def snr(x, xhat):
    """信噪比 SNR = 10 log10( Σx² / Σ(x−x̂)² )."""
    num = float(np.sum(x ** 2))
    den = float(np.sum((x - xhat) ** 2))
    if den <= 1e-300:
        return float("inf")
    return 10.0 * np.log10(num / den)


def mae(x, xhat):
    return float(np.mean(np.abs(x - xhat)))


def detection_stats(true_mask, cand_mask):
    """检测统计: 真噪声中被检出比例 TPR, 干净像素中被误判比例 FPR"""
    tp = int(np.sum(true_mask & cand_mask))
    fn = int(np.sum(true_mask & ~cand_mask))
    fp = int(np.sum(~true_mask & cand_mask))
    tn = int(np.sum(~true_mask & ~cand_mask))
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, tpr=tpr, fpr=fpr,
                n_cand=int(np.sum(cand_mask)))
