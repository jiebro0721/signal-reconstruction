"""问题三: 非线性共轭梯度法 (修正 PRP+) 与 GPSR-BCQP 求解器.

(1) nonlinear_cg: x_{k+1} = x_k + α_k d_k,
    d_k = −g_0 (k=0);  d_k = −g_k + β_k d_{k−1} (k≥1),
    β_k^PRP+ = max{0, g_kᵀy_{k−1}/‖g_{k−1}‖²}, y_{k−1}=g_k−g_{k−1},
    强 Wolfe 线搜索 (c1=1e-4, c2=0.1, scipy 实现), 非下降方向时重启为负梯度。
    可选共轭参数: prp+ / fr / hs+ / dy (文献[5,6,8]对照)。

(2) gpsr_bcqp: GPSR-BB 工程实现 (BCQP + BB1 + 闭式 λ)。
    z=[u;v], x=u−v, min ½zᵀBz+cᵀz s.t. z≥0;
    B=[AᵀA,−AᵀA;−AᵀA,AᵀA], c=τ1+[−Aᵀb;Aᵀb];
    迭代: w=(z−α∇F)_+, λ=mid(0, sᵀg/sᵀBs, 1), z+=z+λ(w−z);
    终止: ‖min(z,∇F(z))‖≤tolP (文献[1]式(17)); 可选去偏置 (去偏置 CG)。
"""
import time

import numpy as np
from scipy.optimize import line_search


# ----------------------------------------------------------------------
# 非线性共轭梯度法
# ----------------------------------------------------------------------

BETAS = {"prp+": "prp+", "prp": "prp", "fr": "fr", "hs+": "hs+", "dy": "dy"}


def beta_rule(rule, g, g_prev, d_prev, y_prev):
    gn = float(np.dot(g, g))
    gy = float(np.dot(g, y_prev))
    if rule == "prp":
        den = float(np.dot(g_prev, g_prev))
        return gy / den if den > 1e-300 else 0.0
    if rule == "prp+":
        den = float(np.dot(g_prev, g_prev))
        b = gy / den if den > 1e-300 else 0.0
        return max(0.0, b)
    if rule == "fr":
        den = float(np.dot(g_prev, g_prev))
        return gn / den if den > 1e-300 else 0.0
    if rule == "hs+":
        den = float(np.dot(d_prev, y_prev))
        b = gy / den if abs(den) > 1e-300 else 0.0
        return max(0.0, b)
    if rule == "dy":
        den = float(np.dot(d_prev, y_prev))
        return gn / den if abs(den) > 1e-300 else 0.0
    raise ValueError(rule)


def strong_wolfe_alpha(model, x, d, g, c1=1e-4, c2=0.1):
    """强 Wolfe 线搜索 (scipy.optimize.line_search 的 zoom 实现):
       f(x+αd) ≤ f(x) + c1 α gᵀd;  |g(x+αd)ᵀd| ≤ c2 |gᵀd|.
    """
    gd = float(np.dot(g, d))
    alpha, _, _, _, _, _ = line_search(
        lambda xx: model.value(xx),
        lambda xx: model.gradient(xx),
        x, d, gfk=g, c1=c1, c2=c2, amax=50.0)
    if alpha is None:
        # 兜底: 回溯 Armijo
        a = 1.0
        f0 = model.value(x)
        for _ in range(40):
            if model.value(x + a * d) <= f0 + c1 * a * gd:
                return float(a), True
            a *= 0.5
        return 0.0, True
    return float(alpha), False


def nonlinear_cg(model, x0, beta="prp+", maxit=5000, tolG=1e-6,
                 c1=1e-4, c2=0.1, stop="grad", rel_tol=1e-6, safeguard=True,
                 fallback=True, track=False):
    """非线性 CG 求解光滑目标 (强 Wolfe). 返回 (x, 统计).

    stop = "grad" : 梯度无穷范数 ≤ tolG 停止;
    stop = "rel"  : 相邻两步目标相对变化 ≤ rel_tol 停止 (文献[3,4,5]准则);
    safeguard=True 时, 方向下降性不足则重启为负梯度;
    fallback=True 时, 线搜索失败则改用负梯度重试 (工程兜底);
    原始 PRP 对照应与 PRP+ 使用相同 safeguard/fallback，仅切换 beta="prp"。
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    g = model.gradient(x)
    d = -g.copy()
    f_cur = model.value(x)
    hist_f = [f_cur]
    hist_g = [float(np.max(np.abs(g)))]
    negative_beta_count = 0
    beta_truncation_count = 0
    descent_restart_count = 0
    line_search_fallback_count = 0
    t0 = time.perf_counter()
    it = 0
    for it in range(1, maxit + 1):
        if float(np.max(np.abs(g))) <= tolG:
            break
        alpha, ls_failed = strong_wolfe_alpha(model, x, d, g, c1, c2)
        line_search_fallback_count += int(ls_failed)
        if alpha <= 0.0 or ls_failed:
            if not fallback:
                break                      # 经典 PRP: 方向不下降时线搜索失败即停滞
            d = -g.copy()
            alpha, ls_failed = strong_wolfe_alpha(model, x, d, g, c1, c2)
            line_search_fallback_count += int(ls_failed)
            if alpha <= 0.0:
                break
        x_new = x + alpha * d
        g_new = model.gradient(x_new)
        f_new = model.value(x_new)
        if stop == "rel" and it > 1:
            rel = abs(f_new - f_cur) / max(abs(f_cur), 1e-12)
            if rel <= rel_tol:
                x, g = x_new, g_new
                hist_f.append(f_new)
                hist_g.append(float(np.max(np.abs(g))))
                break
        y_prev = g_new - g
        if beta in ("prp", "prp+"):
            den = float(np.dot(g, g))
            raw_beta = float(np.dot(g_new, y_prev)) / den if den > 1e-300 else 0.0
        elif beta == "hs+":
            den = float(np.dot(d, y_prev))
            raw_beta = float(np.dot(g_new, y_prev)) / den if abs(den) > 1e-300 else 0.0
        else:
            raw_beta = None
        if raw_beta is not None and raw_beta < 0.0:
            negative_beta_count += 1
            if beta in ("prp+", "hs+"):
                beta_truncation_count += 1
        b = beta_rule(beta, g_new, g, d, y_prev)
        d_new = -g_new + b * d
        gd = float(np.dot(g_new, d_new))
        if safeguard and gd >= -1e-7 * float(np.dot(g_new, g_new)):
            d_new = -g_new.copy()
            descent_restart_count += 1
        x, g, d, f_cur = x_new, g_new, d_new, f_new
        hist_f.append(f_cur)
        hist_g.append(float(np.max(np.abs(g))))
    elapsed = time.perf_counter() - t0
    conv = (stop == "grad" and float(np.max(np.abs(g))) <= tolG) or \
           (stop == "rel" and len(hist_f) >= 2 and
            abs(hist_f[-1] - hist_f[-2]) / max(abs(hist_f[-2]), 1e-12) <= rel_tol)
    return dict(x=x, it=it, elapsed=elapsed,
                hist_f=np.array(hist_f), hist_g=np.array(hist_g),
                conv=bool(conv), negative_beta_count=negative_beta_count,
                beta_truncation_count=beta_truncation_count,
                descent_restart_count=descent_restart_count,
                line_search_fallback_count=line_search_fallback_count)


def cg_with_mu_continuation(model, x0, mu_seq=(1e-1, 1e-2, 1e-3),
                            beta="prp+", maxit=2000, tolG=1e-6,
                            stop="rel", rel_tol=1e-6, safeguard=True, **kw):
    """μ 续延: 由大到小逐段求解, 每段热启动 (文献[4] 的 μ←0.1μ 方案).

    每段终止默认采用相对目标变化 ≤ rel_tol (文献[3,4,5]准则), 避免小 μ 段
    因梯度准则过严而病态。
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    it_sum = 0
    it_hist = []
    stage_diagnostics = []
    conv_all = True
    for mu in mu_seq:
        model.set_mu(float(mu))
        res = nonlinear_cg(model, x, beta=beta, maxit=maxit, tolG=tolG,
                           stop=stop, rel_tol=rel_tol, safeguard=safeguard, **kw)
        x = res["x"]
        it_sum += int(res["it"])
        it_hist.append(int(res["it"]))
        stage_diagnostics.append({
            key: int(res[key]) for key in (
                "negative_beta_count", "beta_truncation_count",
                "descent_restart_count", "line_search_fallback_count"
            )
        })
        conv_all &= bool(res["conv"])
    totals = {
        key: sum(stage[key] for stage in stage_diagnostics)
        for key in (
            "negative_beta_count", "beta_truncation_count",
            "descent_restart_count", "line_search_fallback_count"
        )
    }
    return dict(x=x, it_sum=it_sum, it_hist=it_hist, conv=conv_all,
                stage_diagnostics=stage_diagnostics, **totals)


# ----------------------------------------------------------------------
# GPSR-BB (BCQP 工程实现，并保留文献投影曲率复核模式)
# ----------------------------------------------------------------------

class GpsrBCQP:
    """min ½zᵀBz+cᵀz s.t. z≥0, x = u−v, z=[u;v] (文献[1]式(8)(9))."""

    def __init__(self, A, b, tau):
        self.A = np.asarray(A, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.tau = float(tau)
        self.c0 = self.A.T @ self.b
        self.n = self.A.shape[1]

    def z_to_x(self, z):
        return z[:self.n] - z[self.n:]

    def value(self, z):
        x = self.z_to_x(z)
        r = self.A @ x - self.b
        return float(0.5 * np.dot(r, r) + self.tau * np.sum(z))

    def gradient(self, z):
        x = self.z_to_x(z)
        atr = self.A.T @ (self.A @ x - self.b)
        return np.concatenate([atr, -atr]) + self.tau

    def Bz(self, z):
        x = self.z_to_x(z)
        t = self.A.T @ (self.A @ x)
        return np.concatenate([t, -t])

    def sBs(self, s):
        xs = self.z_to_x(s)
        t = self.A @ xs
        return float(np.dot(t, t))


def projected_bb_stepsize(s, sBs, alpha_min=1e-10, alpha_max=1e6):
    """GPSR-BB spectral step from the projected direction (文献[1] Algorithm 2)."""
    s = np.asarray(s, dtype=np.float64)
    ss = float(np.dot(s, s))
    sBs = float(sBs)
    if sBs <= 0.0 or abs(sBs) < 1e-300:
        return float(alpha_max)
    return float(np.clip(ss / sBs, alpha_min, alpha_max))


def gpsr_bb_solve(A, b, tau, z0=None, maxit=2000, tolP=1e-3,
                  alpha_min=1e-10, alpha_max=1e6, do_debias=False,
                  spectral_update="secant"):
    """GPSR-BB: BB1 + 投影 + 闭式 λ + (可选)去偏置.

    spectral_update="secant" 保留仓库原有的实际步长梯度差更新；
    "projected" 使用文献 Algorithm 2 的完整投影方向曲率更新，供复核对照。
    """
    if spectral_update not in ("secant", "projected"):
        raise ValueError("spectral_update must be 'secant' or 'projected'")
    q = GpsrBCQP(A, b, tau)
    z = np.zeros(2 * q.n) if z0 is None else np.asarray(z0, dtype=np.float64).copy()
    g = q.gradient(z)
    a = min(max(1.0 / max(1.0, float(np.max(np.abs(g)))), alpha_min), alpha_max)
    hist_f = [q.value(z)]
    hist_gap = [float(np.linalg.norm(np.minimum(z, g)))]
    t0 = time.perf_counter()
    it = 0
    for it in range(1, maxit + 1):
        w = np.maximum(z - a * g, 0.0)
        s = w - z
        gs = float(np.dot(g, s))
        ss_Bs = q.sBs(s)
        lam = 1.0
        if ss_Bs > 1e-300:
            # 二次函数沿投影弧的精确最小化: λ* = −(gᵀs)/(sᵀBs) (文献[1]式(16)前的闭式)
            lam = float(np.clip(-gs / ss_Bs, 0.0, 1.0))
        z_new = z + lam * s
        g_new = q.gradient(z_new)
        if spectral_update == "projected":
            # 文献[1] Algorithm 2: 完整投影方向 s 的曲率更新。
            a = projected_bb_stepsize(s, ss_Bs, alpha_min, alpha_max)
        else:
            # 原仓库工程配置: 用实际梯度差作 BB1 secant 更新。
            sy = float(np.dot(s, g_new - g))
            ss = float(np.dot(s, s))
            if sy <= 0.0 or abs(sy) < 1e-300:
                a = alpha_max
            else:
                a = float(np.clip(ss / sy, alpha_min, alpha_max))
        z, g = z_new, g_new
        gap = float(np.linalg.norm(np.minimum(z, g)))
        hist_f.append(q.value(z))
        hist_gap.append(gap)
        if gap <= tolP:
            break
    elapsed = time.perf_counter() - t0
    x = q.z_to_x(z)
    if do_debias:
        x = debias(A, b, x)
    return dict(x=x, z=z, it=it, elapsed=elapsed,
                hist_f=np.array(hist_f), hist_gap=np.array(hist_gap),
                conv=bool(np.linalg.norm(np.minimum(z, q.gradient(z))) <= tolP))


def debias(A, b, x_support, tolD=1e-10):
    """去偏置 (文献[1] §III-D): 固定支撑集, 最小二乘求解该支撑上的最优值."""
    from scipy.sparse import linalg as spla
    amax = float(np.max(np.abs(x_support)))
    idx = np.where(np.abs(x_support) > 1e-3 * max(1.0, amax))[0]
    res = np.zeros_like(x_support)
    if idx.size == 0:
        return res
    As = A[:, idx]
    res[idx] = spla.lsqr(As, b, atol=tolD, btol=tolD)[0]
    return res
