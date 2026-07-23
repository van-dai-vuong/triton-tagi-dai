"""
Linear (fully connected) layer for TAGI.

Forward pass
    μ_z = μ_a @ μ_w  +  μ_b                           (cuBLAS matmul)
    S_z = μ_a² @ S_w  +  S_a @ μ_w²  +  S_a @ S_w  +  S_b   (fused Triton)

Backward pass
    1. Compute weight/bias deltas and store them on the layer
    2. Propagate deltas to the previous layer
    (Parameter update is done separately by the network via the general
     update_parameters function — NOT inside this layer.)

Delta computation (matches cuTAGI):
    Δ_μ_w = S_w · (ma^T @ δ_μ_z)         mean delta
    Δ_S_w = S_w² · ((ma²)^T @ δ_S_z)     variance delta
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..base import LearnableLayer
from ..kernels.common import triton_fused_backward_delta, triton_fused_var_forward
from ..param_init import init_weight_bias_linear
from ..update.parameters import update_parameters


class Linear(LearnableLayer):
    """
    Bayesian fully-connected layer with Gaussian weight distributions.

    Parameters
    ----------
    in_features  : int   number of input neurons
    out_features : int   number of output neurons
    device       : str or torch.device
    init_method  : str   "He" or "Xavier" (default "He")
    gain_w       : float  gain multiplier for weight variance (default 1.0)
    gain_b       : float  gain multiplier for bias variance (default 1.0)
    bias         : bool   whether to include a bias term (default True)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: str = "cpu",
        init_method: str = "He",
        gain_w: float = 1.0,
        gain_b: float = 1.0,
        bias: bool = True,
        generator=None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.device = torch.device(device)

        # --- cuTAGI-style initialization ---
        self.has_bias = bias
        self.mw, self.Sw, self.mb, self.Sb = init_weight_bias_linear(
            in_features,
            out_features,
            init_method=init_method,
            gain_w=gain_w,
            gain_b=gain_b,
            bias=bias,
            device=self.device,
            generator=generator,
        )

        # Saved for backward
        self.ma_in = None

        # Parameter deltas (computed during backward, applied during update)
        self.delta_mw = None
        self.delta_Sw = None
        self.delta_mb = None
        self.delta_Sb = None

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]:
        """
        Propagate Gaussian moments through z = a W + b.

        μ_z = μ_a @ μ_w + μ_b
        S_z = μ_a² @ S_w  +  S_a @ (μ_w² + S_w)  +  S_b

        Accepts any leading batch shape; the last dim must equal ``in_features``.
        Shapes of (B, in) and (B, S, in) are both valid; output mirrors the
        leading dims with ``out_features`` in the last position.
        """
        self._input_shape = ma.shape  # cache for backward reshape
        ma_flat = ma.reshape(-1, self.in_features).contiguous()
        Sa_flat = Sa.reshape(-1, self.in_features).contiguous()
        self.ma_in = ma_flat

        mz_flat = torch.matmul(ma_flat, self.mw) + self.mb
        Sz_flat = triton_fused_var_forward(ma_flat, Sa_flat, self.mw, self.Sw, self.Sb)

        out_shape = (*self._input_shape[:-1], self.out_features)
        return mz_flat.reshape(out_shape), Sz_flat.reshape(out_shape)

    # ------------------------------------------------------------------
    #  Backward (compute deltas only — NO parameter update)
    # ------------------------------------------------------------------
    def backward(self, delta_mz: Tensor, delta_Sz: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute parameter deltas and back-propagate innovation deltas.

        Parameter deltas (cuTAGI convention):
            Δμ_w = S_w · (μ_a^T @ δμ_z)       Δμ_b = S_b · Σ δμ_z
            ΔS_w = S_w² · (μ_a²)^T @ δS_z)    ΔS_b = S_b² · Σ δS_z

        Propagated deltas:
            δμ_a = δμ_z @ μ_w^T
            δS_a = δS_z @ (μ_w²)^T

        Deltas are stored on the layer (self.delta_mw, etc.) but NOT applied.
        Call update() to apply them with capping.

        Parameters
        ----------
        delta_mz : Tensor (B, out_features)  mean delta from next layer
        delta_Sz : Tensor (B, out_features)  variance delta from next layer

        Returns
        -------
        delta_ma : Tensor (B, in_features)   mean delta to propagate
        delta_Sa : Tensor (B, in_features)   variance delta to propagate
        """
        # Flatten any leading batch dims so the matmul / Triton kernels see 2D.
        dmz_flat = delta_mz.reshape(-1, self.out_features).contiguous()
        dSz_flat = delta_Sz.reshape(-1, self.out_features).contiguous()

        # ── Raw gradients (sum over batch) ──
        grad_mw = torch.matmul(self.ma_in.T, dmz_flat)
        grad_mb = dmz_flat.sum(0, keepdim=True)
        grad_Sw = torch.matmul((self.ma_in**2).T, dSz_flat)
        grad_Sb = dSz_flat.sum(0, keepdim=True)

        # ── Parameter deltas (cuTAGI convention) ──
        #   Δ_μ_w = S_w · grad_μ       (prior variance × gradient)
        #   Δ_S_w = S_w² · grad_S      (prior variance² × gradient)
        self.delta_mw = self.Sw * grad_mw
        self.delta_Sw = (self.Sw**2) * grad_Sw

        if self.has_bias:
            self.delta_mb = self.Sb * grad_mb
            self.delta_Sb = (self.Sb**2) * grad_Sb

        # ── Propagate deltas to previous layer ──
        delta_ma_flat, delta_Sa_flat = triton_fused_backward_delta(dmz_flat, dSz_flat, self.mw)
        in_shape = (*self._input_shape[:-1], self.in_features)
        return delta_ma_flat.reshape(in_shape), delta_Sa_flat.reshape(in_shape)

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
        """Total learnable scalars: 2 × (weight + bias) means and variances."""
        n = self.mw.numel() + (self.mb.numel() if self.has_bias else 0)
        return 2 * n

    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features}, bias={self.has_bias})"
