"""
Bayesian RMS Normalization for TAGI.

Normalises the last (feature) dimension of each token by the RMS statistic
including input variance, then applies a learnable Gaussian scale ``γ``
(no bias). Implements the exact cuTAGI formulation (feat/attn-debug).

Forward (per row r over the feature dim of size ``ni``):
    rms[r]      = mean_i( mu_a[r, i]^2 + var_a[r, i] )
    inv_rms[r]  = 1 / sqrt(rms[r] + eps)
    mz[r, i]    = inv_rms[r] * mu_a[r, i] * mw[i]
    Sz[r, i]    = inv_rms[r]^2 * ( Sa[r, i] * (mw[i]^2 + Sw[i])
                                   + Sw[i] * mu_a[r, i]^2 )

Backward (diagonal Jacobian, matching cuTAGI):
    tmp[r, i]         = inv_rms[r] * mw[i]
    delta_ma[r, i]    = tmp[r, i]   * delta_mz[r, i]
    delta_Sa[r, i]    = tmp[r, i]^2 * delta_Sz[r, i]

Weight deltas (scale γ):
    sum_mu[i]  = sum_r ( inv_rms[r]   * mu_a[r, i]   * delta_mz[r, i] )
    sum_var[i] = sum_r ( inv_rms[r]^2 * mu_a[r, i]^2 * delta_Sz[r, i] )
    delta_mw[i]  = sum_mu[i]  * Sw[i]
    delta_Sw[i]  = sum_var[i] * Sw[i]^2

Accepts inputs of shape ``(B, D)`` or ``(B, S, D)``; the leading dims are
flattened for the row-reduction and reshaped on the way out.

Reference: cuTAGI src/rmsnorm_layer.cpp.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..base import LearnableLayer
from ..param_init import init_weight_bias_norm
from ..update.parameters import update_parameters


class RMSNorm(LearnableLayer):
    """Root-mean-square normalisation with Gaussian scale.

    Parameters
    ----------
    normalized_shape : int or list[int]   size of the feature dim
                                          (list of length 1 is also accepted)
    eps              : float              stability epsilon (default 1e-6)
    gain_w           : float              gain multiplier for Sw init (default 1.0)
    device           : str or torch.device
    """

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-6,
        gain_w: float = 1.0,
        device: str = "cpu",
    ) -> None:
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        if len(normalized_shape) != 1:
            raise ValueError(
                f"RMSNorm only supports a single normalized dimension, got {normalized_shape}"
            )
        self.normalized_shape = list(normalized_shape)
        self.ni = normalized_shape[0]
        self.eps = eps
        self.gain_w = gain_w
        self.device = torch.device(device)
        self.has_bias = False

        # cuTAGI init: mu_gamma = 1.0, var_gamma = (gain_w^2) / ni.
        mu_w, Sw, _, _ = init_weight_bias_norm(self.ni, gain_w=gain_w, gain_b=1.0, device=self.device)
        self.mw = mu_w
        self.Sw = Sw
        self.mb = None
        self.Sb = None

        self.ma_in: Tensor | None = None
        self.inv_rms: Tensor | None = None
        self._input_shape: tuple | None = None

        self.delta_mw: Tensor | None = None
        self.delta_Sw: Tensor | None = None
        self.delta_mb = None
        self.delta_Sb = None

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]:
        """Normalise the last dim of ``ma`` by the RMS and apply ``γ``."""
        self._input_shape = ma.shape
        ma_flat = ma.reshape(-1, self.ni)
        Sa_flat = Sa.reshape(-1, self.ni)

        rms = (ma_flat * ma_flat + Sa_flat).mean(dim=1, keepdim=True)
        inv_rms = 1.0 / torch.sqrt(rms + self.eps)

        mz = inv_rms * ma_flat * self.mw
        Sz = (inv_rms * inv_rms) * (
            Sa_flat * (self.mw * self.mw + self.Sw) + self.Sw * ma_flat * ma_flat
        )

        self.ma_in = ma_flat
        self.inv_rms = inv_rms

        return mz.reshape(self._input_shape), Sz.reshape(self._input_shape)

    # ------------------------------------------------------------------
    #  Backward
    # ------------------------------------------------------------------
    def backward(self, delta_mz: Tensor, delta_Sz: Tensor) -> tuple[Tensor, Tensor]:
        """Propagate deltas through the RMS-normalised affine and compute γ grads."""
        shape = delta_mz.shape
        dm = delta_mz.reshape(-1, self.ni)
        dS = delta_Sz.reshape(-1, self.ni)
        inv_rms = self.inv_rms

        # --- Propagated deltas (diagonal Jacobian) ---
        tmp = inv_rms * self.mw
        delta_ma = tmp * dm
        delta_Sa = tmp * tmp * dS

        # --- Scale (γ) deltas: reduce over all rows ---
        ma_in = self.ma_in
        sum_mu = (inv_rms * ma_in * dm).sum(dim=0)
        sum_var = (inv_rms * inv_rms * ma_in * ma_in * dS).sum(dim=0)
        self.delta_mw = sum_mu * self.Sw
        self.delta_Sw = sum_var * self.Sw * self.Sw

        return delta_ma.reshape(shape), delta_Sa.reshape(shape)

    # ------------------------------------------------------------------
    #  Update
    # ------------------------------------------------------------------
    def update(self, cap_factor: float) -> None:
        update_parameters(self.mw, self.Sw, self.delta_mw, self.delta_Sw, cap_factor)

    @property
    def num_parameters(self) -> int:
        return 2 * self.mw.numel()

    def __repr__(self) -> str:
        return (
            f"RMSNorm(normalized_shape={self.normalized_shape}, "
            f"eps={self.eps}, gain_w={self.gain_w})"
        )
