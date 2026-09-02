"""投影梯度求解器(文献[1] GPSR 框架) + 光滑参数续延策略.

问题: 盒约束光滑凸问题
        min_{l ≤ z ≤ h} F_μ(z),   [l, h] = [s_min, s_max]  (最大值原理, 文献[3]命题2)
投影梯度迭代:
        w = P(z − α ∇F_μ(z)),  z ← z + λ (w − z),  λ ∈ [0,1]
    · GPSR-Basic 型: α 在沿投影弧的 Armijo 回溯中收缩(最速下降式), 无 BB 更新;
    · GPSR-BB 型:   α 取 Barzilai-Borwein 步长 sᵀs/sᵀy (截断到 [α_min, α_max]),
                    线搜索 λ 用非单调 Armijo。
                    说明: 文献[1]对二次 BCQP 用闭式 λ 且报告其单调变体略优于
                    非单调变体; 本项目的目标函数非二次, 非单调 Armijo 是
                    通用问题的工程选择(与文献[1]的非单调变体同思路)。
终止: GPSR 投影间隙 ||P(z−∇F) − z||_∞ ≤ tolP, 或 ||Δz||_∞ ≤ tolX / 目标相对变化 ≤ tolF。
"""
import numpy as np
import time


def project(z, lo, hi):
    return np.clip(z, lo, hi)


def projection_gap(z, g, lo, hi):
    """投影间隙 ∞-范数: ‖P(z−g) − z‖_∞, 为 0 当且仅当 z 满足 KKT 条件."""
    return float(np.max(np.abs(np.clip(z - g, lo, hi) - z)))


def bb_stepsize(s, y, amin=1e-10, amax=1e6, variant=1):
    """Barzilai-Borwein 步长: BB1 = sᵀs/sᵀy, BB2 = sᵀy/yᵀy.

    与文献[1]一致: sᵀy ≤ 0 (曲率信息失效) 时取 α_max。
    """
    sy = float(np.dot(s, y))
    ss = float(np.dot(s, s))
    yy = float(np.dot(y, y))
    if sy <= 0.0 or abs(sy) < 1e-300:
        return float(amax)
    if variant == 1:
        a = ss / sy
    else:
        a = sy / yy if abs(yy) > 1e-300 else amax
    return float(np.clip(a, amin, amax))


def gpsr_basic(model, u0, mu=1e-3, lo=0.0, hi=255.0,
               tolP=1e-2, tolX=1e-10, maxit=5000, delta=1e-4, tau=0.5,
               almax_step=20, track=False):
    """GPSR-Basic 推广: 沿投影弧的 Armijo 型回溯(在 α 上回溯, 文献[1]§III-A).

    每步: w = P(z − α∇F(z)); 若 F(w) − F(z) ≤ −δ ∇Fᵀ(z)(w−z) 则接受,
          否则 α ← τ α 重算 w。
    """
    model._set_mu(mu)
    z = np.asarray(u0, dtype=np.float64).copy()
    g = model.gradient(z)
    a = 1.0 / max(1.0, float(np.max(np.abs(g))))
    hist_f = [model.value(z)]
    hist_gap = [projection_gap(z, g, lo, hi)]
    t0 = time.perf_counter()
    converged = False
    for it in range(1, maxit + 1):
        fz = hist_f[-1]
        for _ in range(almax_step):
            w = project(z - a * g, lo, hi)
            d = w - z
            gd = float(np.dot(g, d))
            fw = model.value(w)
            if fw <= fz + delta * gd:
                break
            a *= tau
        else:
            break
        z_new = w
        g_new = model.gradient(z_new)
        gap = projection_gap(z_new, g_new, lo, hi)
        hist_f.append(model.value(z_new))
        hist_gap.append(gap)
        if gap <= tolP or float(np.max(np.abs(z_new - z))) <= tolX:
            converged = True
            z, g = z_new, g_new
            break
        z, g = z_new, g_new
    elapsed = time.perf_counter() - t0
    return dict(u=z, it=it, hist_f=np.array(hist_f), hist_gap=np.array(hist_gap),
                converged=bool(converged), time=elapsed)


def gpsr_bb(model, u0, mu=1e-3, lo=0.0, hi=255.0,
            tolP=1e-2, tolX=1e-10, maxit=5000, amin=1e-10, amax=1e6,
            delta=1e-4, tau=0.5, lsearch_max=30, M=10, bb_variant=1):
    """GPSR-BB 推广: BB 步长 + 投影 + 非单调 Armijo 线搜索(文献[1] §III-B).

    非单调线搜索: F(z+λd) ≤ max_{0≤j≤M} F(z_{k-j}) + δ λ gᵀd,  λ = 1, τ, τ², ...
    """
    model._set_mu(mu)
    z = np.asarray(u0, dtype=np.float64).copy()
    g = model.gradient(z)
    a = 1.0 / max(1.0, float(np.max(np.abs(g))))
    a = float(np.clip(a, amin, amax))
    hist_f = [model.value(z)]
    hist_gap = [projection_gap(z, g, lo, hi)]
    t0 = time.perf_counter()
    converged = False
    for it in range(1, maxit + 1):
        w = project(z - a * g, lo, hi)
        d = w - z
        gd = float(np.dot(g, d))
        fref = max(hist_f[-min(M, len(hist_f)):])   # 非单调参考值
        lam = 1.0
        accepted = False
        for _ in range(lsearch_max):
            fnew = model.value(z + lam * d)
            if fnew <= fref + delta * lam * gd:
                accepted = True
                break
            lam *= tau
        if not accepted:
            # 连续失败: 以最速下降方向兜底
            d = -g / max(1.0, float(np.linalg.norm(g)))
            gd = float(np.dot(g, d))
            for _ in range(lsearch_max):
                fnew = model.value(z + lam * d)
                if fnew <= fref + delta * lam * gd:
                    accepted = True
                    break
                lam *= tau
            if not accepted:
                break
        z_new = z + lam * d
        g_new = model.gradient(z_new)
        s = z_new - z
        y_ = g_new - g
        if float(np.linalg.norm(s)) > 1e-14:
            a = bb_stepsize(s, y_, amin, amax, variant=bb_variant)
        z, g = z_new, g_new
        gap = projection_gap(z, g, lo, hi)
        hist_f.append(model.value(z))
        hist_gap.append(gap)
        if gap <= tolP or float(np.max(np.abs(s))) <= tolX:
            converged = True
            break
    elapsed = time.perf_counter() - t0
    return dict(u=z, it=it, hist_f=np.array(hist_f), hist_gap=np.array(hist_gap),
                converged=bool(converged), time=elapsed)


def solve_with_continuation(model, u0, mu_seq=(1e-1, 1e-2, 1e-3),
                            tolP=1e-2, maxit_each=3000, solver=gpsr_bb, **kw):
    """光滑参数 μ 的续延策略(文献[4,9]): 由大到小逐段求解, 每段以上一段解热启动."""
    z = np.asarray(u0, dtype=np.float64).copy()
    it_sum = 0
    it_hist = []
    conv_all = True
    for mu in mu_seq:
        res = solver(model, z, mu=float(mu), tolP=tolP, maxit=maxit_each, **kw)
        z = res["u"]
        it_sum += int(res["it"])
        it_hist.append(int(res["it"]))
        conv_all &= bool(res["converged"])
    return dict(u=z, it_sum=it_sum, it_hist=it_hist, converged=conv_all)
