"""
EvenSoftplus activation layer for TAGI-V (heteroscedastic noise learning).

In TAGI-V, the output layer has 2×K outputs:
    - Even indices (0, 2, 4, ...): mean predictions → passed through unchanged
    - Odd indices  (1, 3, 5, ...): log-variance predictions → softplus applied

The softplus function  f(z) = log(1 + exp(z))  restricts variance predictions
to the positive domain.

Forward pass (for odd indices only):
    Using a second-order Gaussian approximation:
        σ(μ_z) = sigmoid(μ_z)
        μ_a ≈ softplus(μ_z) + 0.5 · S_z · σ(μ_z) · (1 − σ(μ_z))
        J   = σ(μ_z)               (derivative of softplus = sigmoid)
        S_a = J² · S_z

Even indices pass through unchanged (identity).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..base import Layer

EPS = 1e-9


# ======================================================================
#  Python API
# ======================================================================


def even_softplus(mz, Sz, half_width):
    """
    Apply softplus to odd-indexed positions, identity to even-indexed.

    Layout: for each batch item, positions are [mean_0, var_0, mean_1, var_1, ...]
    Odd positions (1, 3, 5, ...) get softplus; even (0, 2, 4, ...) are identity.
    Inputs are processed flat, so an element's parity is its flat-index parity.

    Parameters
    ----------
    mz         : Tensor (flat)  pre-activation means
    Sz         : Tensor (flat)  pre-activation variances
    half_width : int            K = number of observation dimensions (unused;
                                kept for signature compatibility)

    Returns
    -------
    ma : Tensor  post-activation means
    Sa : Tensor  post-activation variances
    J  : Tensor  Jacobian (sigmoid for odd, 1.0 for even)
    """
    is_odd = (torch.arange(mz.numel(), device=mz.device) % 2) == 1

    # ── Softplus moments (for odd positions) ──
    sig = torch.sigmoid(mz)
    softplus_base = torch.where(mz > 20.0, mz, torch.log1p(torch.exp(mz)))

    # Second-order correction to the mean
    mu_sp = (softplus_base + 0.5 * Sz * sig * (1.0 - sig)).clamp_min(EPS)

    # Jacobian = sigmoid(mz); variance = J² · S_z
    J_sp = sig
    Sa_sp = (J_sp * J_sp * Sz).clamp_min(EPS)

    # ── Select: odd → softplus, even → identity ──
    ma = torch.where(is_odd, mu_sp, mz)
    Sa = torch.where(is_odd, Sa_sp, Sz)
    J = torch.where(is_odd, J_sp, torch.ones_like(J_sp))

    return ma, Sa, J


# ======================================================================
#  EvenSoftplus Layer
# ======================================================================


class EvenSoftplus(Layer):
    """
    Activation layer for TAGI-V heteroscedastic output.

    Applies softplus to odd-indexed positions (variance predictions)
    and passes even-indexed positions (mean predictions) through unchanged.

    The output dimension must be 2×K where K is the number of target dimensions.

    Parameters
    ----------
    half_width : int  K = number of observation dimensions (e.g., 10 for CIFAR-10)
    """

    def __init__(self, half_width: int) -> None:
        self.half_width = half_width
        self.J = None  # stored Jacobian

    def forward(self, mz: Tensor, Sz: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        mz : Tensor (B, 2K)  pre-activation means
        Sz : Tensor (B, 2K)  pre-activation variances

        Returns
        -------
        ma : Tensor (B, 2K)  post-activation means
        Sa : Tensor (B, 2K)  post-activation variances
        """
        original_shape = mz.shape
        ma, Sa, J = even_softplus(mz.reshape(-1), Sz.reshape(-1), self.half_width)
        self.J = J.view(original_shape)
        return ma.view(original_shape), Sa.view(original_shape)

    def backward(self, delta_ma: Tensor, delta_Sa: Tensor) -> tuple[Tensor, Tensor]:
        """
        Propagate deltas back through the EvenSoftplus layer.

        For even positions (identity): delta passes through unchanged.
        For odd positions (softplus):  delta_mz = J · delta_ma
                                       delta_Sz = J² · delta_Sa

        Parameters
        ----------
        delta_ma : Tensor (B, 2K)  mean delta in activation space
        delta_Sa : Tensor (B, 2K)  variance delta in activation space

        Returns
        -------
        delta_mz : Tensor (B, 2K)  mean delta in pre-activation space
        delta_Sz : Tensor (B, 2K)  variance delta in pre-activation space
        """
        J = self.J
        return delta_ma * J, delta_Sa * J * J

    def __repr__(self):
        return f"EvenSoftplus(half_width={self.half_width})"
