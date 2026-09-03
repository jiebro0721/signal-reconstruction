"""第一阶段: 自适应中值滤波器 (AMF) —— 文献[2] Algorithm I 的向量化实现。

对每个像素 (i,j)，窗口从 w=3 开始, 若窗口内 s_min < s_med < s_max 则停止扩大,
否则 w += 2 直至 w_max(按噪声等级查文献[2]表1)。随后(严格按文献步骤 3-5):
    - 若某级窗口满足 s_min < s_med < s_max: 用该窗口检查 s_min < y_ij < s_max,
      成立则判为未污染像素, 否则判为噪声候选(输出为该窗口中值);
    - 若窗口耗尽(w 达 w_max 仍不满足): 无条件判为噪声候选, 输出 w_max 窗口中值。

边界像素: 采用反射 (reflect) 扩展, 等价于对称边界条件, 影响很小。
"""
import numpy as np
from scipy import ndimage

# 文献[2]表1: 不同噪声等级 r 对应的最大窗口 w_max (窗口为 w_max x w_max)
WMAX_TABLE = [
    (0.25, 5), (0.40, 7), (0.60, 9), (0.70, 13),
    (0.80, 17), (0.85, 25), (0.90, 39), (float('inf'), 39),
]


def w_max_for_noise_level(r):
    for thr, w in WMAX_TABLE:
        if r < thr or thr == float('inf'):
            return w
    return 39


def _window_stats(y, w, mode="reflect"):
    """对每个像素计算 w×w 窗口内的 min/median/max."""
    # 步长 1 的滑动窗口, 尺寸 w×w
    h = ndimage.minimum_filter(y, size=w, mode=mode)
    m = ndimage.median_filter(y, size=w, mode=mode)
    M = ndimage.maximum_filter(y, size=w, mode=mode)
    return h, m, M


def adaptive_median_filter(y, r=0.3, wmax=None, s_min=0.0, s_max=255.0,
                           verbose=False):
    """自适应中值滤波检测噪声候选.

    参数
    ----
    y      : 污染图像
    r      : 噪声等级(用于 w_max 查表)
    wmax   : 最大窗口; None 时按文献[2]表1由 r 确定
    s_min, s_max : 像素动态范围端点

    返回
    ----
    cand :  bool ndarray, True = 噪声候选集 N
    y_amf : ndarray, AMF 输出(候选像素被中值替换, 其余保持原值), 用于第二阶段初值
    stats : dict, 每个像素最终窗口的 (s_min, s_med, s_max)
    """
    y = np.asarray(y, dtype=np.float64)
    M, N = y.shape
    if wmax is None:
        wmax = w_max_for_noise_level(r)

    # 逐级扩大窗口: 记录每个像素"首次满足 smin<smed<smax 的窗口统计"
    # done: 已经满足条件(或 w>wmax 直接判为候选)的像素
    smin = np.full((M, N), np.nan)
    smed = np.full((M, N), np.nan)
    smax = np.full((M, N), np.nan)
    satisfied = np.zeros((M, N), dtype=bool)   # 是否通过 smin<smed<smax 判定
    active = np.ones((M, N), dtype=bool)       # 尚未确定最终窗口的像素

    for w in range(3, wmax + 1, 2):
        idx = np.where(active)
        if len(idx[0]) == 0:
            break
        lo, med, hi = _window_stats(y, w, mode="reflect")
        # 当前仍活跃的像素: 检查条件 s_min < s_med < s_max
        chk = active & (lo < med) & (med < hi)
        smin[chk], smed[chk], smax[chk] = lo[chk], med[chk], hi[chk]
        satisfied |= chk
        # 条件满足即停止扩窗; 还有多数值窗口(窗口内全是同一极值)则继续
        active = active & ~chk
        # w 继续增大

    # 剩余(所有 w 都不满足 smin<smed<smax)的像素: 用 wmax 窗口统计, 直接判为候选
    if np.any(active):
        lo, med, hi = _window_stats(y, wmax, mode="reflect")
        smin[active], smed[active], smax[active] = lo[active], med[active], hi[active]

    # 判定(严格按文献[2] Algorithm I 步骤 3-5 与 Algorithm III 步骤 1):
    #   - 步骤3: 若某级窗口满足 smin<smed<smax, 转步骤5 —— 用该窗口(min,max)检查 y;
    #   - 步骤4: 窗口耗尽(w 超过 w_max 仍不满足) —— 无条件判为噪声候选并替换为中值;
    #   - 步骤5: s_min < y < s_max 则非候选, 否则为候选;
    #   - Algorithm III 步骤1: 噪声候选集 N = {(i,j): y_ij ∈ {s_min,s_max} 且 ŷ_ij ≠ y_ij},
    #     即只把"取值为端点值且被 AMF 替换过"的像素判为候选(噪声像素取值恰为端点值,
    #     未发生替换说明窗口无信息, 将其留给第二阶段的数据保真项处理亦可, 但按文献
    #     定义从候选集中排除, 避免把平坦区间内的同值干净像素误判为噪声)。
    exhausted = active.copy()               # 从未获得满足窗口的像素(窗口耗尽)
    amf_cand = exhausted | ~((smin < y) & (y < smax))
    y_amf = y.copy()
    y_amf[amf_cand] = smed[amf_cand]
    # 文献[2] Algorithm III 步骤 1 的候选集定义:
    #   N = {(i,j): y_ij ∈ {s_min,s_max} 且 ŷ_ij ≠ y_ij}
    # 即噪声像素取值恰为动态范围端点, 且 AMF 输出了替换值(中值不等于原值)。
    cand = ((y == s_min) | (y == s_max)) & (y_amf != y)
    return cand, y_amf, dict(smin=smin, smed=smed, smax=smax)
