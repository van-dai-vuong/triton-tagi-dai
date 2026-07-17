"""
Remax activation layer for TAGI.

Remax = rectified-Gaussian softmax alternative for Bayesian networks.

Given logits Z ~ N(μ_z, S_z), computes output probabilities A = M / Σ M
where M_k = max(0, Z_k) is the rectified Gaussian.

Forward (cuTAGI parity — L. Alric, 2024):
    1. MixtureReLU moments of M = max(0, Z)
    2. Log-normal moments of M and M̃ = Σ M_k
    3. cov(ln M, ln M̃) via log-normal identity
    4. A = exp(ln M - ln M̃); renormalize so Σ μ_A = 1
    5. cov(A, M) via log-normal identity
    6. cov(A, Z) = cov(A, M) / Φ(α)   (capped by Cauchy–Schwarz)
    7. Jacobian  J = cov(A, Z) / var(Z)
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import triton
import triton.language as tl
from torch import Tensor

from ..base import Layer

# ======================================================================
#  Triton kernel — Remax (MixtureReLU + log-normal path, cuTAGI parity)
# ======================================================================


@triton.jit
def _remax_kernel(
    mu_z_ptr,
    var_z_ptr,            # (B, K) inputs
    mu_a_ptr,
    var_a_ptr,            # (B, K) outputs
    J_ptr,                # (B, K) Jacobian cov(A,Z)/var(Z)
    K,                    # number of classes (runtime)
    stride_b,
    BLOCK_K: tl.constexpr,
):
    """One program ↔ one batch item.  All K classes in registers."""
    b = tl.program_id(0)
    base = b * stride_b
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    EPS: tl.constexpr = 1e-6              # match cuTAGI's floor
    INV_SQRT_2PI: tl.constexpr = 0.3989422804014327
    INV_SQRT_2:   tl.constexpr = 0.7071067811865475

    # ── load Z moments ──
    mu_z = tl.load(mu_z_ptr + base + offs, mask=mask, other=0.0)
    # other=1.0 for var_z so masked lanes don't pollute sums / divisions
    var_z = tl.load(var_z_ptr + base + offs, mask=mask, other=1.0)

    # ── 1. MixtureReLU moments  M = max(0, Z)  (L. Alric, 2024) ──
    std_z = tl.sqrt(var_z)
    alpha = mu_z / std_z
    pdf_alpha = INV_SQRT_2PI * tl.exp(-0.5 * alpha * alpha)
    cdf_alpha = 0.5 + 0.5 * tl.math.erf(alpha * INV_SQRT_2)

    mu_m = mu_z * cdf_alpha + std_z * pdf_alpha
    mu_m = tl.maximum(mu_m, EPS)

    var_m = (
        -mu_m * mu_m
        + 2.0 * mu_m * mu_z
        - mu_z * std_z * pdf_alpha
        + (var_z - mu_z * mu_z) * cdf_alpha
    )
    var_m = tl.maximum(var_m, EPS)
    jcb_m = cdf_alpha                      # = Φ(α), used later

    # ── 2. Sum moments M̃ = Σ M_k ──
    mu_mt  = tl.maximum(tl.sum(tl.where(mask, mu_m,  0.0)), EPS)
    var_mt = tl.maximum(tl.sum(tl.where(mask, var_m, 0.0)), EPS)

    # ── 3. Log-normal moments of M and M̃ ──
    var_log_m  = tl.log(1.0 + var_m  / (mu_m  * mu_m))
    mu_log_m   = tl.log(mu_m)  - 0.5 * var_log_m
    var_log_mt = tl.log(1.0 + var_mt / (mu_mt * mu_mt))
    mu_log_mt  = tl.log(mu_mt) - 0.5 * var_log_mt

    # ── 4. cov(ln M_k, ln M̃) ──
    cov_log_m_mt = tl.log(1.0 + var_m / (mu_m * mu_mt))

    # ── 5. Moments of ln(A) ──
    mu_log_a  = mu_log_m - mu_log_mt
    var_log_a = tl.maximum(var_log_m + var_log_mt - 2.0 * cov_log_m_mt, 0.0)

    # ── 6. Remax probabilities (renormalized) ──
    mu_a_raw = tl.maximum(tl.exp(mu_log_a + 0.5 * var_log_a), EPS)
    sum_mu_a = tl.maximum(tl.sum(tl.where(mask, mu_a_raw, 0.0)), EPS)
    mu_a     = mu_a_raw / sum_mu_a
    var_a    = (tl.exp(var_log_a) - 1.0) * mu_a * mu_a

    # ── 7. Jacobian J = cov(A, Z) / var(Z) ──
    # cov(ln A, ln M) = var_log_m - cov_log_m_mt
    # cov(A, M) via log-normal identity: (exp(·) - 1) · μ_A · μ_M
    # cov(A, Z) = cov(A, M) / Φ(α), capped by Cauchy–Schwarz bound
    cov_log_a_log_m = var_log_m - cov_log_m_mt
    cov_a_m = (tl.exp(cov_log_a_log_m) - 1.0) * mu_a * mu_m

    cs_bound = tl.sqrt(var_a * var_z)
    cov_a_z  = tl.minimum(cs_bound, cov_a_m / tl.maximum(jcb_m, EPS))
    J        = cov_a_z / var_z

    # ── store ──
    tl.store(mu_a_ptr + base + offs, mu_a, mask=mask)
    tl.store(var_a_ptr + base + offs, var_a, mask=mask)
    tl.store(J_ptr     + base + offs, J,    mask=mask)


# ======================================================================
#  Python wrapper
# ======================================================================


def triton_remax(mu_z: Tensor, var_z: Tensor):
    """
    Compute Remax moments and Jacobian using fused Triton kernel.

    Parameters
    ----------
    mu_z  : Tensor (B, K)  logit means
    var_z : Tensor (B, K)  logit variances

    Returns
    -------
    mu_a  : Tensor (B, K)  probability means (Σ μ_a ≈ 1)
    var_a : Tensor (B, K)  probability variances
    J     : Tensor (B, K)  Jacobian  J = cov(A, Z) / var(Z)
    """
    squeeze = mu_z.dim() == 1
    if squeeze:
        mu_z = mu_z.unsqueeze(0)
        var_z = var_z.unsqueeze(0)

    mu_z = mu_z.contiguous()
    var_z = var_z.contiguous()
    B, K = mu_z.shape

    mu_a  = torch.empty_like(mu_z)
    var_a = torch.empty_like(mu_z)
    J     = torch.empty_like(mu_z)

    BLOCK_K = triton.next_power_of_2(K)
    _remax_kernel[(B,)](
        mu_z, var_z, mu_a, var_a, J,
        K, mu_z.stride(0),
        BLOCK_K=BLOCK_K,
    )

    if squeeze:
        return mu_a.squeeze(0), var_a.squeeze(0), J.squeeze(0)
    return mu_a, var_a, J


@lru_cache(maxsize=16)
def _legendre_unit_interval(num_quad: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(num_quad)
    return (0.5 * (x + 1.0), 0.5 * w)


def _stable_cdf_scaled(
    alpha: Tensor,
    beta: Tensor,
    mu: Tensor,
    var: Tensor,
    t: Tensor,
) -> Tensor:
    """Return exp(-mu*t + 0.5*var*t^2) * Phi(beta)."""
    inv_sqrt2 = 0.7071067811865475
    expo = -mu * t + 0.5 * var * t * t
    log_q = expo + torch.special.log_ndtr(beta)
    q_log = torch.exp(torch.clamp(log_q, min=-745.0, max=709.0))

    erfcx_arg = torch.clamp(-beta * inv_sqrt2, min=0.0)
    q_erfcx = 0.5 * torch.exp(-0.5 * alpha * alpha) * torch.special.erfcx(erfcx_arg)
    return torch.where(beta < -10.0, q_erfcx, q_log)


def laplace_remax(
    mu_z: Tensor,
    var_z: Tensor,
    *,
    num_quad: int = 48,
    full_jacobian: bool = True,
    cap_jacobian: bool = True,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Laplace-Remax moments for independent Gaussian logits.

    This path uses fixed quadrature over the Laplace identity for ``1 / S`` and
    ``1 / S^2``. It is intended for small output dimensions such as CIFAR-10.
    ``J[b, i, j]`` stores ``Cov(A_i, Z_j) / Var(Z_j)`` when
    ``full_jacobian=True``; otherwise the diagonal is returned.
    """
    if num_quad < 2:
        raise ValueError(f"num_quad must be >= 2, got {num_quad}")

    squeeze = mu_z.dim() == 1
    original_shape = mu_z.shape
    if squeeze:
        mu_z = mu_z.unsqueeze(0)
        var_z = var_z.unsqueeze(0)
    elif mu_z.dim() > 2:
        mu_z = mu_z.reshape(-1, mu_z.shape[-1])
        var_z = var_z.reshape(-1, var_z.shape[-1])

    out_dtype = mu_z.dtype
    device = mu_z.device
    work_dtype = torch.float64
    mu = mu_z.to(work_dtype)
    var = var_z.clamp_min(eps).to(work_dtype)
    B, K = mu.shape

    u_np, w_np = _legendre_unit_interval(num_quad)
    u = torch.as_tensor(u_np, dtype=work_dtype, device=device).view(1, num_quad, 1)
    w = torch.as_tensor(w_np, dtype=work_dtype, device=device).view(1, num_quad, 1)
    one_minus_u = (1.0 - u).clamp_min(eps)
    t = u / one_minus_u
    quad_w = w / (one_minus_u * one_minus_u)

    mu_q = mu.unsqueeze(1)
    var_q = var.unsqueeze(1)
    sigma = torch.sqrt(var_q)
    alpha = mu_q / sigma
    beta = alpha - sigma * t

    inv_sqrt_2pi = 0.3989422804014327
    phi_alpha = inv_sqrt_2pi * torch.exp(-0.5 * alpha * alpha)
    p0 = torch.special.ndtr(-alpha)
    q = _stable_cdf_scaled(alpha, beta, mu_q, var_q, t)

    L = (p0 + q).clamp_min(eps)
    z_shift = mu_q - var_q * t
    D = z_shift * q + sigma * phi_alpha
    F = (z_shift * z_shift + var_q) * q + z_shift * sigma * phi_alpha
    N = mu_q * p0 - sigma * phi_alpha
    H = N + D

    log_P = torch.sum(torch.log(L), dim=-1, keepdim=True)
    P = torch.exp(log_P)
    P_over_L = P / L

    mu_a = torch.sum(quad_w * D * P_over_L, dim=1)
    second_diag = torch.sum(quad_w * t * F * P_over_L, dim=1)

    pi0 = torch.prod(p0.squeeze(1), dim=-1, keepdim=True)
    mu_a = mu_a + pi0 / K
    second_diag = second_diag + pi0 / (K * K)

    R_D = D / L
    R_H = H / L
    EAZ = torch.einsum(
        "bq,bqi,bqj->bij",
        (quad_w.squeeze(-1) * P.squeeze(-1)),
        R_D,
        R_H,
    )
    diag_EAZ = torch.sum(quad_w * F * P_over_L, dim=1)
    diag_idx = torch.arange(K, device=device)
    EAZ[:, diag_idx, diag_idx] = diag_EAZ

    p0_flat = p0.squeeze(1).clamp_min(torch.finfo(work_dtype).tiny)
    product_except = torch.exp(
        torch.sum(torch.log(p0_flat), dim=-1, keepdim=True) - torch.log(p0_flat)
    )
    EAZ = EAZ + (N.squeeze(1).unsqueeze(1) * product_except.unsqueeze(1)) / K

    var_a = (second_diag - mu_a * mu_a).clamp_min(0.0)
    cov_az = EAZ - mu_a.unsqueeze(2) * mu.unsqueeze(1)
    if cap_jacobian:
        bound = torch.sqrt(var_a.unsqueeze(2) * var.unsqueeze(1)).clamp_min(0.0)
        cov_az = torch.clamp(cov_az, min=-bound, max=bound)
    J_full = cov_az / var.unsqueeze(1).clamp_min(eps)
    J = J_full if full_jacobian else torch.diagonal(J_full, dim1=1, dim2=2)

    mu_a = mu_a.to(out_dtype).reshape(original_shape)
    var_a = var_a.to(out_dtype).reshape(original_shape)
    if full_jacobian:
        if squeeze:
            J = J.squeeze(0).to(out_dtype)
        elif len(original_shape) > 2:
            J = J.to(out_dtype).reshape(*original_shape[:-1], K, K)
        else:
            J = J.to(out_dtype)
    else:
        J = J.to(out_dtype).reshape(original_shape)

    return mu_a, var_a, J


# ======================================================================
#  Remax Layer
# ======================================================================


class Remax(Layer):
    """
    Remax activation layer (softmax alternative for Bayesian networks).

    Forward:  (μ_z, S_z) → (μ_a, S_a)  with Σ μ_a ≈ 1
    Backward: uses J = cov(Z, A) / var(Z) as Jacobian

    Stores J from the forward pass for use during backward.
    """

    def __init__(
        self,
        approximation: str = "lognormal",
        jacobian: str = "diag",
        num_quad: int = 48,
        cap_jacobian: bool = True,
    ) -> None:
        if approximation not in {"lognormal", "laplace"}:
            raise ValueError(
                f"approximation must be 'lognormal' or 'laplace', got {approximation!r}"
            )
        if jacobian not in {"diag", "full"}:
            raise ValueError(f"jacobian must be 'diag' or 'full', got {jacobian!r}")
        if approximation == "lognormal" and jacobian != "diag":
            raise ValueError("lognormal Remax only exposes a diagonal Jacobian")
        self.approximation = approximation
        self.jacobian = jacobian
        self.num_quad = num_quad
        self.cap_jacobian = cap_jacobian
        self.J: Tensor | None = None  # stored Jacobian

    def forward(self, mz: Tensor, Sz: Tensor) -> tuple[Tensor, Tensor]:
        if self.approximation == "laplace":
            mu_a, Sa, J = laplace_remax(
                mz,
                Sz,
                num_quad=self.num_quad,
                full_jacobian=self.jacobian == "full",
                cap_jacobian=self.cap_jacobian,
            )
        else:
            mu_a, Sa, J = triton_remax(mz, Sz)
        self.J = J
        return mu_a, Sa

    def backward(self, delta_ma: Tensor, delta_Sa: Tensor) -> tuple[Tensor, Tensor]:
        """Propagate deltas through a_k ≈ μ_a_k + J_k · (z_k − μ_z_k)."""
        J = self.J
        if J is None:
            raise RuntimeError("Remax.backward called before forward")
        if J.dim() == delta_ma.dim() + 1:
            return (
                torch.einsum("...i,...ij->...j", delta_ma, J),
                torch.einsum("...i,...ij->...j", delta_Sa, J * J),
            )
        return delta_ma * J, delta_Sa * J * J

    def __repr__(self) -> str:
        if self.approximation == "lognormal":
            return "Remax()"
        return (
            "Remax(approximation='laplace', "
            f"jacobian='{self.jacobian}', num_quad={self.num_quad})"
        )
