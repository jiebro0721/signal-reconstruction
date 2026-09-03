"""问题三: 稀疏信号重构 —— 数据生成 + 光滑化 ℓ1-二次模型 (文献[1] §IV-A 复现).

实例: n=4096, k=1024, 160 个 ±1 尖峰, y = Ax + e, σ²=1e-4,
      A: k×n 高斯矩阵行正交化, τ = 0.1‖Aᵀy‖∞ (文献[1]式(22)).
模型:  min_x τ‖x‖₁ + ½‖Ax−b‖²  → 光滑化  min_x F_μ(x)=τ Σ ρ_μ(x_i) + ½‖Ax−b‖²,
      ρ_μ(t)=√(μ²+t²), |ρ_μ−|·|| ≤ μ;  ∇F_μ = τ ρ_μ'(x) + Aᵀ(Ax−b).
"""
import numpy as np

from restoration_model import SMOOTHINGS


# ----------------------------------------------------------------------
# 数据生成
# ----------------------------------------------------------------------

def gen_sparse_signal(n=4096, k_true=160, rng=None):
    """长度为 n 的 k_true 稀疏信号: 非零位置随机, 取值 ±1."""
    rng = np.random.default_rng() if rng is None else rng
    x = np.zeros(n)
    idx = rng.choice(n, size=k_true, replace=False)
    x[idx] = rng.choice([-1.0, 1.0], size=k_true)
    return x


def gen_measurement(m=1024, n=4096, rng=None):
    """m×n 高斯随机矩阵, 行正交化 (与文献[1]相同: A = Q' from qr(randn(n,m)))."""
    rng = np.random.default_rng() if rng is None else rng
    G = rng.standard_normal((n, m))
    Q, _ = np.linalg.qr(G, mode="reduced")
    return Q.T


def make_instance(m=1024, n=4096, k_true=160, sigma2=1e-4, seed=0):
    """生成完整实例: (A, x_true, b, sigma2)."""
    rng = np.random.default_rng(seed)
    A = gen_measurement(m, n, rng)
    x = gen_sparse_signal(n, k_true, rng)
    b = A @ x + np.sqrt(sigma2) * rng.standard_normal(m)
    return A, x, b


def tau_gpsr(A, b):
    """文献[1]式(22): τ = 0.1 ‖Aᵀy‖∞."""
    return 0.1 * float(np.max(np.abs(A.T @ b)))


# ----------------------------------------------------------------------
# 光滑化 ℓ1-二次目标
# ----------------------------------------------------------------------

class SmoothedL1Quad:
    """F_μ(x) = τ Σ ρ_μ(x_i) + ½‖Ax−b‖²."""

    def __init__(self, A, b, tau, mu=1e-3, smooth="sqrt"):
        self.A = np.asarray(A, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.tau = float(tau)
        self.rho, self.drho = SMOOTHINGS[smooth]
        self.mu = float(mu)
        self.At = self.A.T
        self.AtA_x = None            # 缓存

    def value(self, x):
        r = self.A @ x - self.b
        return float(self.tau * np.sum(self.rho(x, self.mu)) + 0.5 * np.dot(r, r))

    def gradient(self, x):
        r = self.A @ x - self.b
        return self.tau * self.drho(x, self.mu) + self.At @ r

    def predict(self, x):
        return self.A @ x

    def set_mu(self, mu):
        self.mu = float(mu)


def finite_diff_check(model, x, eps=1e-6, n_sample=6, seed=0):
    g = model.gradient(x)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(g), size=min(n_sample, len(g)), replace=False)
    err = 0.0
    for i in idx:
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        gfd = (model.value(xp) - model.value(xm)) / (2 * eps)
        err = max(err, abs(g[i] - gfd) / max(1e-12, abs(gfd)))
    return float(err)
