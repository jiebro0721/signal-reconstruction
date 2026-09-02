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


# ---------------------------------------------------------------------------
# 结构保持类指标 (问题二用): SSIM 与边缘 PSNR
# ---------------------------------------------------------------------------

def ssim(x, xhat, data_range=255.0, win=11, sigma=1.5, K1=0.01, K2=0.03):
    """标准 SSIM (高斯窗口). 返回全图均值. x, xhat 为 2-D."""
    from scipy import ndimage
    x = np.asarray(x, dtype=np.float64)
    xh = np.asarray(xhat, dtype=np.float64)
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    g = np.outer(np.exp(-0.5 * (np.arange(win) - win // 2) ** 2 / sigma ** 2),
                 np.exp(-0.5 * (np.arange(win) - win // 2) ** 2 / sigma ** 2))
    g /= g.sum()
    mu1 = ndimage.convolve(x, g, mode="reflect")
    mu2 = ndimage.convolve(xh, g, mode="reflect")
    s11 = ndimage.convolve(x * x, g, mode="reflect") - mu1 ** 2
    s22 = ndimage.convolve(xh * xh, g, mode="reflect") - mu2 ** 2
    s12 = ndimage.convolve(x * xh, g, mode="reflect") - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))
    return float(np.mean(ssim_map))


def edge_mask(x, thresh=80.0, dilate=2):
    """Sobel 边缘掩膜: 梯度模长 > thresh 的像素(膨胀 dilate 像素)."""
    from scipy import ndimage
    x = np.asarray(x, dtype=np.float64)
    sx = ndimage.sobel(x, axis=0)
    sy = ndimage.sobel(x, axis=1)
    mag = np.hypot(sx, sy)
    m = mag > thresh
    if dilate > 0:
        m = ndimage.binary_dilation(m, iterations=dilate)
    return m


def edge_psnr(x, xhat, thresh=80.0, dilate=2, peak=255.0):
    """边缘区域 PSNR (衡量边缘/细节保持)."""
    m = edge_mask(x, thresh, dilate)
    if not np.any(m):
        return float("inf")
    mse = float(np.mean((x[m] - xhat[m]) ** 2))
    return 10.0 * np.log10(peak * peak / max(mse, 1e-300))
