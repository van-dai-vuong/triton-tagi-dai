"""
Bayesian ReLU activation layer for TAGI.

Computes the exact moments of a = max(0, z) where z ~ N(μ_z, S_z).

Using α = μ_z / σ_z :
    μ_a = σ_z · φ(α)  +  μ_z · Φ(α)
    S_a = −μ_a² + 2μ_a·μ_z − μ_z·σ_z·φ(α) + (S_z − μ_z²)·Φ(α)
    J   = Φ(α)     (Jacobian = CDF, used during backward)

where φ and Φ are the standard normal PDF and CDF respectively.

Numerical notes
---------------
var_a is floored at 0 to keep variance non-negative; mu_a has no floor,
matching the exact cuTAGI mixture_relu_mean_var_cuda formula.

CDF precision: the standard formula 0.5*(1+erf(α/√2)) suffers catastrophic
cancellation for α ≤ −6 in fp32, returning exactly 0 when the true value is
~10⁻⁹.  cuTAGI's CUDA backend uses erfcf(−α/√2) which avoids this.  We switch
to the asymptotic expansion Φ(α) ≈ φ(α)/|α| for α < −5, which is both more
accurate and exactly what CUDA's erfcf computes internally for large arguments.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..base import Layer

INV_SQRT_2PI = 0.3989422804014327  # 1/√(2π)
INV_SQRT_2 = 0.7071067811865476  # 1/√2


# ======================================================================
#  Python API
# ======================================================================


def bayesian_relu(mz, Sz):
    """
    Compute exact moments of a rectified Gaussian.

    Parameters
    ----------
    mz : Tensor  pre-activation means   (any shape, elementwise)
    Sz : Tensor  pre-activation variances

    Returns
    -------
    ma : Tensor  post-activation means
    Sa : Tensor  post-activation variances
    J  : Tensor  Jacobian = Φ(α), used during backward
    """
    # Protect sqrt against exact-zero Sz (harmless in practice; matches cuTAGI
    # where std_z = sqrtf(var_z) with no guard since var_z > 0 after init).
    Sz_safe = Sz.clamp_min(1e-30)
    sigma_z = torch.sqrt(Sz_safe)
    alpha = mz / sigma_z

    pdf = torch.exp(-0.5 * alpha * alpha) * INV_SQRT_2PI

    # CDF: standard erf formula underflows to exactly 0.0 in fp32 for alpha ≤ -6,
    # while cuTAGI's CUDA erfcf correctly returns ~Φ(alpha) ≈ 1e-9.  For alpha < -5
    # the asymptotic expansion Φ(α) ≈ φ(α)/|α| is both more accurate and avoids
    # the cancellation artefact (and reuses pdf at zero extra cost).
    cdf_erf = 0.5 * (1.0 + torch.erf(alpha * INV_SQRT_2))
    cdf_asym = pdf / (-alpha)  # accurate for alpha ≪ 0; pdf/|α| > 0 always
    cdf = torch.where(alpha < -5.0, cdf_asym, cdf_erf)

    # Exact moments matching cuTAGI mixture_relu_mean_var_cuda.
    mu_a = sigma_z * pdf + mz * cdf
    var_a = -mu_a * mu_a + 2.0 * mu_a * mz - mz * sigma_z * pdf + (Sz_safe - mz * mz) * cdf
    var_a = var_a.clamp_min(0.0)

    return mu_a, var_a, cdf


class ReLU(Layer):
    """
    Bayesian ReLU activation layer.

    Stores the Jacobian from the forward pass for use during backward.
    """

    def __init__(self) -> None:
        self.J = None  # stored Jacobian

    def forward(self, mz: Tensor, Sz: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        mz : Tensor (B, D)  pre-activation means
        Sz : Tensor (B, D)  pre-activation variances

        Returns
        -------
        ma : Tensor (B, D)  post-activation means
        Sa : Tensor (B, D)  post-activation variances
        """
        original_shape = mz.shape
        ma, Sa, J = bayesian_relu(mz.reshape(-1), Sz.reshape(-1))
        self.J = J.view(original_shape)
        return ma.view(original_shape), Sa.view(original_shape)

    def backward(self, delta_mz: Tensor, delta_Sz: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply stored Jacobian to the deltas.

        J for mean, J² for variance (chain rule on Gaussian moments).

        Parameters
        ----------
        delta_mz : Tensor (B, D)  mean delta
        delta_Sz : Tensor (B, D)  variance delta

        Returns
        -------
        delta_ma : Tensor (B, D)  mean delta to propagate
        delta_Sa : Tensor (B, D)  variance delta to propagate
        """
        J = self.J
        return delta_mz * J, delta_Sz * J * J

    def __repr__(self):
        return "ReLU()"
