"""
Bayesian AvgPool2D layer for TAGI.

Average pooling for means:
    μ_out = (1/k²) · Σ μ_in

Variance pooling depends on `spatial_correlation`:
    - False (default): S_out = (1/k⁴) · Σ S_in
          assumes pixel independence (strict Bayesian); variance collapses by
          1/k² at each pool.  Matches cuTAGI's historical behaviour.
    - True:  S_out = (1/k²) · Σ S_in
          assumes the k×k window is strongly correlated (ρ≈1); preserves the
          magnitude of the variance signal through pooling bottlenecks.
          Experimental — turn on for variance-preservation ablations; net
          init may need retuning since downstream variance is up to k² per
          pool layer larger than in the default branch.

Pure-PyTorch implementation (runs on CPU / CUDA / MPS). Windows are
non-overlapping k×k (stride = k), so pooling reduces to a reshape + sum and
the backward is a nearest-neighbour upsample.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..base import Layer


# ======================================================================
#  AvgPool2D Layer
# ======================================================================


class AvgPool2D(Layer):
    """
    Bayesian average pooling layer.

    Parameters
    ----------
    kernel_size : int   pooling window size (square)
    spatial_correlation : bool, default False
        If True, treat the k×k window as strongly correlated (ρ≈1) and scale the
        output variance by 1/k² instead of 1/k⁴.  Prevents variance starvation
        through pooling layers but up-amplifies downstream variance — nets may
        need retuning.  Default False matches cuTAGI's strict-independence
        behaviour.
    """

    def __init__(self, kernel_size: int, spatial_correlation: bool = False) -> None:
        self.k = kernel_size
        self.spatial_correlation = spatial_correlation
        self.input_shape = None

    def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        ma : Tensor (N, C, H, W)  activation means
        Sa : Tensor (N, C, H, W)  activation variances

        Returns
        -------
        ma_out : Tensor (N, C, H//k, W//k)
        Sa_out : Tensor (N, C, H//k, W//k)
        """
        self.input_shape = ma.shape
        N, C, H, W = ma.shape
        k = self.k
        H_out, W_out = H // k, W // k

        inv_k2 = 1.0 / (k * k)
        var_scale = inv_k2 if self.spatial_correlation else inv_k2 * inv_k2

        # Non-overlapping k×k windows: split H→(H_out, k), W→(W_out, k), sum.
        sum_m = ma.reshape(N, C, H_out, k, W_out, k).sum(dim=(3, 5))
        sum_s = Sa.reshape(N, C, H_out, k, W_out, k).sum(dim=(3, 5))

        return sum_m * inv_k2, sum_s * var_scale

    def backward(self, dm: Tensor, ds: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        dm : Tensor (N, C, H_out, W_out)  mean delta
        ds : Tensor (N, C, H_out, W_out)  variance delta

        Returns
        -------
        dm_out : Tensor (N, C, H, W)
        ds_out : Tensor (N, C, H, W)
        """
        k = self.k
        inv_k2 = 1.0 / (k * k)
        var_scale = inv_k2 if self.spatial_correlation else inv_k2 * inv_k2

        # Each output pixel distributes its delta to the whole k×k input block.
        dm_up = dm.repeat_interleave(k, dim=2).repeat_interleave(k, dim=3)
        ds_up = ds.repeat_interleave(k, dim=2).repeat_interleave(k, dim=3)

        return dm_up * inv_k2, ds_up * var_scale

    def __repr__(self):
        sc = "on" if self.spatial_correlation else "off"
        return f"AvgPool2D(kernel={self.k}, spatial_correlation={sc})"
