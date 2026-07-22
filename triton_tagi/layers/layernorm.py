"""
Bayesian Layer Normalization for TAGI (1-D / MLP mode).

Normalises the feature dimension of each sample independently using sample
statistics, then applies a learnable affine transform with Gaussian
parameters γ (mw/Sw) and β (mb/Sb).

Forward pass (per sample b, per feature i):
    1. Compute per-sample statistics:
           mu_ra[b]  = mean_i(mu_a[b, i])
           var_s[b]  = sum_i(Sa[b, i])
           var_ra[b] = (sum_i((mu_a[b,i] - mu_ra[b])^2) + var_s[b]) / (ni - 1)
       Note: var_ra uses Bessel correction + input-uncertainty term, matching cuTAGI.

    2. Normalise and apply affine:
           inv_std    = 1 / sqrt(var_ra[b] + eps)
           mu_z[b,i]  = inv_std * (mu_a[b,i] - mu_ra[b]) * mw[i] + mb[i]
           var_z[b,i] = inv_std^2 * (Sa[b,i]*(mw[i]^2+Sw[i])
                        + Sw[i]*(mu_a[b,i]-mu_ra[b])^2) + Sb[i]

Backward pass (diagonal Jacobian approximation, matching cuTAGI):
    J[b,i] = inv_std[b] * mw[i]     (ignoring cross-terms from mean subtraction)

    delta_ma[b,i] = J[b,i]   * delta_mz[b,i]
    delta_Sa[b,i] = J[b,i]^2 * delta_Sz[b,i]

Parameter deltas (cuTAGI convention):
    For weight (gamma):
        tmp_w[b,i] = inv_std[b] * (mu_a[b,i] - mu_ra[b]) * Sw[i]
        delta_mw[i] = sum_b( tmp_w[b,i] * delta_mz[b,i] )
        delta_Sw[i] = sum_b( tmp_w[b,i]^2 * delta_Sz[b,i] )
    For bias (beta):
        delta_mb[i] = sum_b( Sb[i] * delta_mz[b,i] )   = Sb[i] * sum_b delta_mz[b,i]
        delta_Sb[i] = sum_b( Sb[i]^2 * delta_Sz[b,i] ) = Sb[i]^2 * sum_b delta_Sz[b,i]
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..base import LearnableLayer
from ..param_init import init_weight_bias_norm
from ..update.parameters import update_parameters


class LayerNorm(LearnableLayer):
    """
    Bayesian Layer Normalization for TAGI (MLP / 1-D feature mode).

    Normalises over the last (feature) dimension of each sample.
    Unlike BatchNorm, there are no running statistics — stats are
    computed fresh per forward call (matching cuTAGI LayerNorm).

    Parameters
    ----------
    normalized_shape : int   number of features (ni)
    eps              : float numerical stability constant (default 1e-5)
    bias             : bool  whether to include a bias term (default True)
    device           : str or torch.device
    gain_w           : float gain multiplier for gamma variance (default 1.0)
    gain_b           : float gain multiplier for beta variance (default 1.0)
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        bias: bool = True,
        device: str = "cpu",
        gain_w: float = 1.0,
        gain_b: float = 1.0,
    ) -> None:
        self.normalized_shape = normalized_shape
        self.ni = normalized_shape
        self.eps = eps
        self.has_bias = bias
        self.device = torch.device(device)

        self.mw, self.Sw, self.mb, self.Sb = init_weight_bias_norm(
            normalized_shape, gain_w=gain_w, gain_b=gain_b, device=self.device
        )
        if not bias:
            self.mb = torch.zeros(normalized_shape, device=self.device)
            self.Sb = torch.zeros(normalized_shape, device=self.device)

        # Cache for backward (populated in forward)
        self.ma_in: Tensor | None = None
        self.mu_ra: Tensor | None = None   # (B, 1) per-sample mean
        self.var_ra: Tensor | None = None  # (B, 1) per-sample variance (Bessel)

        # Parameter deltas (computed in backward, applied in update)
        self.delta_mw: Tensor | None = None
        self.delta_Sw: Tensor | None = None
        self.delta_mb: Tensor | None = None
        self.delta_Sb: Tensor | None = None

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        ma : Tensor (B, ni)   input activation means
        Sa : Tensor (B, ni)   input activation variances

        Returns
        -------
        mz : Tensor (B, ni)   normalized + affine output means
        Sz : Tensor (B, ni)   normalized + affine output variances
        """
        B, ni = ma.shape

        # ── Per-sample statistics ──
        mu_ra = ma.mean(dim=1, keepdim=True)                          # (B, 1)
        var_s = Sa.sum(dim=1, keepdim=True)                           # (B, 1)
        mu_diff = ma - mu_ra                                          # (B, ni)
        var_ra = (mu_diff.pow(2).sum(dim=1, keepdim=True) + var_s) / (ni - 1)  # (B, 1)

        inv_std = 1.0 / torch.sqrt(var_ra + self.eps)                 # (B, 1)

        # ── Normalised means ──
        mz = inv_std * mu_diff * self.mw + self.mb                    # (B, ni)

        # ── Normalised variances ──
        # var_z[b,i] = inv_std[b]^2 * (Sa[b,i]*(mw[i]^2+Sw[i]) + Sw[i]*mu_diff[b,i]^2) + Sb[i]
        Sz = inv_std**2 * (Sa * (self.mw**2 + self.Sw) + self.Sw * mu_diff**2) + self.Sb

        # ── Cache for backward ──
        self.ma_in = ma
        self.mu_ra = mu_ra
        self.var_ra = var_ra

        return mz, Sz

    # ------------------------------------------------------------------
    #  Backward (compute deltas only — NO parameter update)
    # ------------------------------------------------------------------
    def backward(self, delta_mz: Tensor, delta_Sz: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute parameter deltas and propagate to the previous layer.

        Parameters
        ----------
        delta_mz : Tensor (B, ni)  mean delta from next layer
        delta_Sz : Tensor (B, ni)  variance delta from next layer

        Returns
        -------
        delta_ma : Tensor (B, ni)  mean delta to propagate
        delta_Sa : Tensor (B, ni)  variance delta to propagate
        """
        inv_std = 1.0 / torch.sqrt(self.var_ra + self.eps)  # (B, 1)

        # ── Jacobian: J = inv_std * mw (diagonal, one per element) ──
        J = inv_std * self.mw   # (B, ni), broadcast

        # ── Propagate delta to previous layer ──
        delta_ma = J * delta_mz
        delta_Sa = J**2 * delta_Sz

        # ── Parameter deltas ──
        mu_diff = self.ma_in - self.mu_ra  # (B, ni)

        # Weight (gamma): tmp_w = inv_std * mu_diff * Sw
        tmp_w = inv_std * mu_diff * self.Sw        # (B, ni)
        self.delta_mw = (tmp_w * delta_mz).sum(dim=0)       # (ni,)
        self.delta_Sw = (tmp_w**2 * delta_Sz).sum(dim=0)    # (ni,)

        # Bias (beta): delta = Sb * sum_batch(delta)
        self.delta_mb = (self.Sb * delta_mz).sum(dim=0)     # (ni,)
        self.delta_Sb = (self.Sb**2 * delta_Sz).sum(dim=0)  # (ni,)

        return delta_ma, delta_Sa

    # ------------------------------------------------------------------
    #  Update (apply capped deltas — called by the network)
    # ------------------------------------------------------------------
    def update(self, cap_factor: float) -> None:
        """
        Apply the stored parameter deltas with cuTAGI-style capping.

        Parameters
        ----------
        cap_factor : float  regularisation strength (from get_cap_factor)
        """
        update_parameters(self.mw, self.Sw, self.delta_mw, self.delta_Sw, cap_factor)
        if self.has_bias:
            update_parameters(self.mb, self.Sb, self.delta_mb, self.delta_Sb, cap_factor)

    @property
    def num_parameters(self) -> int:
        """Total learnable scalars: 2 × (gamma + beta) means and variances."""
        n = self.mw.numel() + (self.mb.numel() if self.has_bias else 0)
        return 2 * n

    def __repr__(self):
        return (
            f"LayerNorm(normalized_shape={self.normalized_shape}, "
            f"eps={self.eps}, bias={self.has_bias})"
        )
