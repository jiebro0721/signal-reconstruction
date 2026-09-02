"""
噪声模型与图像工具
"""
import numpy as np


def add_salt_and_pepper(x, p=0.15, q=0.15, rng=None, smin=0.0, smax=255.0):
    """在灰度图像 x 上叠加经典 salt-and-pepper 椒盐噪声。

    模型: y_ij = smin (概率 p) / smax (概率 q) / x_ij (其余), 噪声等级 r = p + q.

    参数
    ----
    x : ndarray (M, N) 原始图像
    p, q : 椒(0)、盐(255)噪声概率
    rng : np.random.Generator 随机数生成器
    smin, smax : 动态范围

    返回
    ----
    y : ndarray 污染图像
    mask : ndarray(bool) 真实噪声像素位置 (True = 被污染)
    """
    x = np.asarray(x, dtype=np.float64)
    rng = rng if rng is not None else np.random.default_rng()
    r = rng.random(x.shape)
    mask = (r < p) | (r >= 1.0 - q)          # 前 p 比例->smin, 后 q 比例->smax
    y = x.copy()
    y[r < p] = smin
    y[r >= 1.0 - q] = smax
    return y, mask


def load_gray(path, size=None):
    """读取灰度图像, 转 float64, 可选下采样到 size×size (用于 256×256 图).

    部分标准测试 TIFF 带额外 alpha 通道(PIL 无法识别), 此时用 tifffile 读取首通道。
    """
    from PIL import Image
    try:
        img = Image.open(path).convert("L")
        a = np.asarray(img, dtype=np.float64)
    except Exception:
        import tifffile
        a = tifffile.imread(path)
        if a.ndim == 3:
            a = a[:, :, 0]
        a = np.asarray(a, dtype=np.float64)
    if size is not None:
        from PIL import Image as _I
        a = np.asarray(_I.fromarray(a).resize((size, size), _I.BILINEAR),
                       dtype=np.float64)
    return a
