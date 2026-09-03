"""第二阶段: 保边正则化恢复模型(光滑化版本) —— 目标函数、梯度、光滑函数族。

模型 (问题一)
-------------
给定第一阶段检测出的噪声候选集 Ω (=N), 在 Ω 上最小化

    F_β(u) = Σ_{(i,j)∈N} |u_ij − y_ij| + (β/2)·(S1 + S2),
    S1 = Σ_{(m,n)∈V_ij∩N^C} 2·φ_α(u_ij − y_mn),
    S2 = Σ_{(m,n)∈V_ij∩N}   φ_α(u_ij − u_mn),

其中 V_ij 为 (i,j) 的 4-邻域(不含自身), φ_α 为保边势函数, 本文主选 φ_α(t)=√(t²+α).

非光滑项 |u_ij − y_ij| 用光滑函数 ρ_μ(t) 逼近 (μ 为光滑参数), 得到光滑模型

    min F_μ(u) := Σ_{N} ρ_μ(u_ij − y_ij) + (β/2)(S1 + S2),   μ ↓ 0 时 F_μ → F_β.

等价记法: 令 ũ_p = u_p (p∈N), ũ_p = y_p (p∉N), 则
    F_μ(u) = Σ_{p∈N} ρ_μ(u_p − y_p) + β Σ_{无向边 (p,q): p∈N} φ_α(u_p − ũ_q).
梯度 (已由分部求和验证):
    ∇F_μ(u_p) = ρ_μ′(u_p − y_p) + β Σ_{q∈V_p} φ_α′(u_p − ũ_q).

光滑函数的逼近性质(见文献[4]): |ρ_μ(t) − |t|| ≤ c·μ, ∀t ∈ R.
"""
import numpy as np

# ---------------------------------------------------------------------------
# 光滑函数 ρ_μ(t) ≈ |t|  (文献[4]中的 φ1, φ3, φ4 实现)
# ---------------------------------------------------------------------------

def rho_sqrt(t, mu):
    """ρ(t) = sqrt(mu^2 + t^2),  文献[4] φ3 的变体(φ3 取 4μ^2, 此处恒等缩放)."""
    return np.sqrt(t * t + mu * mu)


def drho_sqrt(t, mu):
    return t / np.sqrt(t * t + mu * mu)


def rho_huber(t, mu):
    """Huber 型: 文献[4] φ4.  C^1 光滑, 有界二阶导. 对 |t| 的逼近误差 ≤ μ/2."""
    a = np.abs(t)
    out = np.where(a <= mu, t * t / (2.0 * mu), a - mu / 2.0)
    return out


def drho_huber(t, mu):
    return np.where(np.abs(t) <= mu, t / mu, np.sign(t))


def rho_softplus(t, mu):
    """ρ(t) = mu*[ln(1+e^{-t/mu}) + ln(1+e^{t/mu})],  文献[4] φ1. 数值稳定实现.

    等价形式: |t| + 2 mu ln(1 + e^{-|t|/mu});  C^1, 逼近误差 sup |ρ−|t|| = 2μ ln2.
    """
    a = np.abs(t)
    return a + 2.0 * mu * np.log1p(np.exp(-a / mu))


def drho_softplus(t, mu):
    return np.tanh(t / (2.0 * mu))


SMOOTHINGS = {"sqrt": (rho_sqrt, drho_sqrt),
              "huber": (rho_huber, drho_huber),
              "softplus": (rho_softplus, drho_softplus)}

# ---------------------------------------------------------------------------
# 保边势函数 φ_α (题目问题二列出的 5 种, 文献[3] eq.(3)-(7))
# 共同性质: 偶、凸、关于 |t| 严格递增, |t| 远离 0 时 φ_α(t) ≈ |t| (按比例/尺度含义)
# 光滑化: 5 种势函数带 (value, deriv) 与 (smooth, smooth_deriv):
#   - 本身光滑的 (√(t²+α)、log cosh、|t|/α−log) 直接使用;
#   - C¹ 的 Huber 直接使用(其二阶导分段有界);
#   - |t|^α (1<α<2) 仅 C¹ 且 φ'' 在 0 附近奇异, 采用文献[4]的 p-范数光滑化
#     (t²+μ²)^{α/2} ⇒ C^∞, 与原函数在 |t|≫μ 时一致, 且消除原点奇异;
#     (文献[4]对 lp 正则的光滑家族 SF(III) 即 (μ̄²+t²)^{(p-1)/2}φ_j(μ,t)
#      的同源技巧, 此处取 φ=√(t²+μ²) 型合并后即 (μ²+t²)^{p/2}。)
# ---------------------------------------------------------------------------

def phi_sqrt(t, alpha=100.0):
    """(1) φ_α(t) = sqrt(t^2 + α),  α > 0; 处处光滑 C^∞."""
    return np.sqrt(t * t + alpha)


def dphi_sqrt(t, alpha=100.0):
    return t / np.sqrt(t * t + alpha)


def phi_sqrt_sm(t, alpha=100.0, mu=1.0):
    return phi_sqrt(t, alpha)


def dphi_sqrt_sm(t, alpha=100.0, mu=1.0):
    return dphi_sqrt(t, alpha)


def phi_power(t, alpha=1.3):
    """(2) φ_α(t) = |t|^α, 1 < α ≤ 2;  C^1 (α<2 时仅 C^1, φ'' 在 0 附近奇异)."""
    return np.abs(t) ** alpha


def dphi_power(t, alpha=1.3):
    return alpha * np.sign(t) * np.abs(t) ** (alpha - 1.0)


def phi_power_sm(t, alpha=1.3, mu=1.0):
    """|t|^α 的光滑化 (t²+μ²)^{α/2}: C^∞, 与原函数在 |t|≫μ 时一致 (文献[4])."""
    return (t * t + mu * mu) ** (alpha / 2.0)


def dphi_power_sm(t, alpha=1.3, mu=1.0):
    return alpha * t * (t * t + mu * mu) ** (alpha / 2.0 - 1.0)


def phi_logcosh(t, alpha=0.1):
    """(3) φ_α(t) = log(cosh(α t)), α > 0;  C^∞. 数值稳定: logcosh(x)=|x|-log2+log1p(e^{-2|x|})."""
    x = alpha * t
    return np.abs(x) - np.log(2.0) + np.log1p(np.exp(-2.0 * np.abs(x)))


def dphi_logcosh(t, alpha=0.1):
    return alpha * np.tanh(alpha * t)


def phi_logcosh_sm(t, alpha=0.1, mu=1.0):
    return phi_logcosh(t, alpha)


def dphi_logcosh_sm(t, alpha=0.1, mu=1.0):
    return dphi_logcosh(t, alpha)


def phi_log1(t, alpha=10.0):
    """(4) φ_α(t) = |t|/α − log(1 + |t|/α),  α > 0;  C^2, 且 |φ_α'| ≤ 1/α (全局 Lipschitz)."""
    a = np.abs(t)
    return a / alpha - np.log1p(a / alpha)


def dphi_log1(t, alpha=10.0):
    return np.sign(t) * (1.0 / alpha - 1.0 / (alpha + np.abs(t)))


def phi_log1_sm(t, alpha=10.0, mu=1.0):
    return phi_log1(t, alpha)


def dphi_log1_sm(t, alpha=10.0, mu=1.0):
    return dphi_log1(t, alpha)


def phi_huber(t, alpha=10.0):
    """(5) Huber 型: φ_α(t) = t²/(2α) (|t|≤α) / |t| − α/2 (|t|>α);  C^1, |φ_α'| ≤ 1."""
    a = np.abs(t)
    return np.where(a <= alpha, t * t / (2.0 * alpha), a - alpha / 2.0)


def dphi_huber(t, alpha=10.0):
    return np.where(np.abs(t) <= alpha, t / alpha, np.sign(t))


def phi_huber_sm(t, alpha=10.0, mu=1.0):
    return phi_huber(t, alpha)


def dphi_huber_sm(t, alpha=10.0, mu=1.0):
    return dphi_huber(t, alpha)


POTENTIALS = {"sqrt": (phi_sqrt, dphi_sqrt, phi_sqrt_sm, dphi_sqrt_sm),
              "power": (phi_power, dphi_power, phi_power_sm, dphi_power_sm),
              "logcosh": (phi_logcosh, dphi_logcosh, phi_logcosh_sm, dphi_logcosh_sm),
              "log1": (phi_log1, dphi_log1, phi_log1_sm, dphi_log1_sm),
              "huber": (phi_huber, dphi_huber, phi_huber_sm, dphi_huber_sm)}

# ---------------------------------------------------------------------------
# 二阶恢复模型 (在噪声候选集 N 上)
# ---------------------------------------------------------------------------

class Phase2Model:
    """建立并评估光滑化后的第二阶段目标函数 F_μ.

    仅以 N 内像素为变量; 拓扑信息(N 与 4-邻域的关系)预计算一次.
    """

    def __init__(self, y, mask, beta=5.0, alpha=100.0,
                 smooth="sqrt", potential="sqrt", boundary="reflect"):
        """
        y         : (M,N) 噪声图像
        mask      : (M,N) bool, True = 噪声候选集 N
        beta      : 正则化参数
        alpha     : 势函数参数 (二次型: 曲率/尺度; power: 指数; logcosh: 速率)
        smooth    : 光滑函数名 (sqrt/huber/softplus)
        potential : 保边势函数名 (sqrt/power/logcosh/log1/huber)
        """
        self.y = np.asarray(y, dtype=np.float64)
        self.mask = np.asarray(mask, dtype=bool)
        self.M, self.N = self.y.shape
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.phi, self.dphi, self.phi_sm, self.dphi_sm = POTENTIALS[potential]
        self.potential = potential
        self.rho, self.drho = SMOOTHINGS[smooth]
        self.smooth = smooth
        self.mu = 1.0             # 光滑参数(主配置 μ=1, 求解器可再显式覆盖)
        self.with_data = True     # 是否保留数据保真项 Σ|u_ij − y_ij| (文献[3]讨论)
        self.nvars = int(np.sum(self.mask))
        self.idx_map = np.full(self.mask.shape, -1, dtype=np.int64)
        self.idx_map[self.mask] = np.arange(self.nvars, dtype=np.int64)
        # 变量顺序对应的像素坐标 (i, j)
        self.coords = np.argwhere(self.mask)
        self._build_topology()

    def _build_topology(self):
        """预计算 4 个方向的邻居信息.

        对每个方向 d, 记录长度为 nvars 的数组:
          nb_kind[d] : 1 = 邻居也在 N 内(变量), 0 = 邻居为干净像素(固定 y), -1 = 越界(忽略)
          nb_idx[d]  : 邻居变量编号 (仅 nb_kind==1 时有意义)
          nb_yval[d] : 邻居的原观测值 (仅 nb_kind==0 时有意义)
        """
        nv = self.nvars
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.ndirs = len(dirs)
        self.nb_kind = np.empty((self.ndirs, nv), dtype=np.int8)
        self.nb_idx = np.empty((self.ndirs, nv), dtype=np.int64)
        self.nb_yval = np.empty((self.ndirs, nv), dtype=np.float64)
        for d, (di, dj) in enumerate(dirs):
            pi, pj = self.coords[:, 0] + di, self.coords[:, 1] + dj
            inb = (pi >= 0) & (pi < self.M) & (pj >= 0) & (pj < self.N)
            kind = np.full(nv, -1, dtype=np.int8)
            idx = np.zeros(nv, dtype=np.int64)
            yv = np.zeros(nv, dtype=np.float64)
            kind[inb] = 0
            yv[inb] = self.y[pi[inb], pj[inb]]
            isn = np.zeros(nv, dtype=bool)
            isn[inb] = self.mask[pi[inb], pj[inb]]
            has = np.where(isn)
            kind[has] = 1
            idx[has] = self.idx_map[pi[has], pj[has]]
            self.nb_kind[d] = kind
            self.nb_idx[d] = idx
            self.nb_yval[d] = yv

    # ------------------------------------------------------------------
    def value(self, u):
        """F_μ(u) = Σ_N ρ_μ(u−y) + (β/2) Σ_N [ Σ_{nbr∉N} 2φ(u−y_nbr) + Σ_{nbr∈N} φ(u−u_nbr) ]"""
        u = np.asarray(u, dtype=np.float64)
        yN = self.y[self.mask]
        fdata = np.sum(self.rho(u - yN, self.mu)) if self.with_data else 0.0
        freg = 0.0
        for d in range(self.ndirs):
            k = self.nb_kind[d]
            m0 = np.where(k == 0)[0]          # 邻居为干净像素: 2φ_μ(u − y)
            if m0.size:
                freg += 2.0 * np.sum(self.phi_sm(u[m0] - self.nb_yval[d][m0],
                                                 self.alpha, self.mu))
            m1 = np.where(k == 1)[0]          # 邻居为噪声候选变量: φ_μ(u − u_nbr)
            if m1.size:
                freg += np.sum(self.phi_sm(u[m1] - u[self.nb_idx[d][m1]],
                                           self.alpha, self.mu))
        return fdata + 0.5 * self.beta * freg

    def gradient(self, u):
        """∇F_μ(u_p) = ρ_μ′(u_p − y_p) + β Σ_{q∈V_p} φ_α′(u_p − ũ_q),  ũ=u(N), y(N^C)."""
        u = np.asarray(u, dtype=np.float64)
        yN = self.y[self.mask]
        g = self.drho(u - yN, self.mu) if self.with_data else np.zeros_like(u)
        for d in range(self.ndirs):
            k = self.nb_kind[d]
            m0 = np.where(k == 0)[0]
            if m0.size:
                g[m0] += self.beta * self.dphi_sm(u[m0] - self.nb_yval[d][m0],
                                                  self.alpha, self.mu)
            m1 = np.where(k == 1)[0]
            if m1.size:
                g[m1] += self.beta * self.dphi_sm(u[m1] - u[self.nb_idx[d][m1]],
                                                  self.alpha, self.mu)
        return g

    def _set_mu(self, mu):
        self.mu = float(mu)


def finite_diff_check(model, u, mu, eps=1e-6, n_sample=6, seed=0):
    """梯度有限差分校验: 随机抽取 n_sample 个分量(返回最大相对误差)."""
    import numpy as np
    model._set_mu(mu)
    g = model.gradient(u)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(g), size=min(n_sample, len(g)), replace=False)
    gfd = np.zeros_like(idx, dtype=np.float64)
    for k, i in enumerate(idx):
        up = u.copy(); up[i] += eps
        um = u.copy(); um[i] -= eps
        gfd[k] = (model.value(up) - model.value(um)) / (2 * eps)
    err = np.max(np.abs(g[idx] - gfd)) / max(1e-12, np.max(np.abs(gfd)))
    return float(err)
