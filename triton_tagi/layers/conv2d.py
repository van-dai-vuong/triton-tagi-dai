"""
Bayesian Conv2D layer for TAGI.

Uses im2col to unfold the input into patch matrices, then leverages
the same fused variance-forward and backward-delta kernels as the
Linear layer (all the heavy lifting stays in Triton).

Forward:
    1. im2col:  (N, C_in, H, W) → (N·L, K)  where K = C_in·kH·kW
    2. Mean:    mz = patches_ma @ mw + mb     (cuBLAS)
    3. Variance: Sz = fused_var_forward(...)   (Triton)
    4. Reshape: (N·L, C_out) → (N, C_out, H_out, W_out)

Backward:
    1. Compute weight/bias deltas (stored, not applied)
    2. Back-project deltas through weights (Triton fused)
    3. col2im: scatter deltas back to spatial layout
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..base import LearnableLayer
from ..kernels.common import triton_fused_backward_delta, triton_fused_var_forward
from ..param_init import init_weight_bias_conv2d
from ..update.parameters import update_parameters


# ======================================================================
#  im2col / col2im (pure PyTorch, via F.unfold / F.fold)
#
#  F.unfold enumerates each sliding block as a column with channel-major then
#  (kh, kw) ordering — exactly the K = c·(kH·kW) + kh·kW + kw layout, and blocks
#  ordered row-major as oh·W_out + ow. F.fold is its exact scatter-add adjoint,
#  matching the original col2im kernel.
# ======================================================================


def _triton_im2col(x, kH, kW, stride, padding):
    """Unfold spatial input into a patch matrix (N·L, K)."""
    N = x.shape[0]
    C = x.shape[1]
    K = C * kH * kW
    # F.unfold -> (N, K, L); reorder to (N·L, K) with L = H_out·W_out.
    cols = F.unfold(x, kernel_size=(kH, kW), stride=stride, padding=padding)
    return cols.transpose(1, 2).reshape(N * cols.shape[2], K)


def _triton_col2im(col, N, C, H, W, kH, kW, stride, padding):
    """Fold a patch matrix (N·L, K) back into spatial layout (N, C, H, W)."""
    K = C * kH * kW
    L = col.shape[0] // N
    # (N·L, K) -> (N, K, L) then scatter-add back to the image.
    cols = col.reshape(N, L, K).transpose(1, 2)
    return F.fold(
        cols,
        output_size=(H, W),
        kernel_size=(kH, kW),
        stride=stride,
        padding=padding,
    )


# ======================================================================
#  Conv2D Layer
# ======================================================================


class Conv2D(LearnableLayer):
    """
    Bayesian Conv2D layer with Gaussian weight distributions.

    Parameters
    ----------
    C_in         : int  input channels
    C_out        : int  output channels (filters)
    kernel_size  : int  square kernel size
    stride       : int  (default 1)
    padding      : int  (default 0)
    padding_type : int  1 = symmetric (PyTorch default), 2 = right-bottom only (cuTAGI style)
    device      : str or torch.device
    init_method : str  "He" or "Xavier" (default "He")
    gain_w      : float  gain multiplier for weight variance (default 1.0)
    gain_b      : float  gain multiplier for bias variance (default 1.0)
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        padding_type: int = 1,
        device: str = "cpu",
        init_method: str = "He",
        gain_w: float = 1.0,
        gain_b: float = 1.0,
    ) -> None:
        self.C_in = C_in
        self.C_out = C_out
        self.kH = self.kW = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_type = padding_type  # 1=symmetric, 2=right-bottom (cuTAGI)
        self.device = torch.device(device)

        # --- cuTAGI-style initialization ---
        self.mw, self.Sw, self.mb, self.Sb = init_weight_bias_conv2d(
            kernel_size,
            C_in,
            C_out,
            init_method=init_method,
            gain_w=gain_w,
            gain_b=gain_b,
            device=self.device,
        )
        self.has_bias = True

        # Stored for backward
        self.patches_ma = None
        self.input_shape = None
        self.spatial = None

        # Parameter deltas
        self.delta_mw = None
        self.delta_Sw = None
        self.delta_mb = None
        self.delta_Sb = None

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]:
        """
        Propagate Gaussian moments through a Conv2D layer via im2col.

        im2col unfolds the input into a patch matrix P of shape (N·L, K)
        where L = H_out·W_out and K = C_in·kH·kW.  The layer then reduces
        to a Linear layer on P:

            μ_z = P_μ @ μ_w + μ_b
            S_z = P_μ² @ S_w  +  P_S @ (μ_w² + S_w)  +  S_b

        Parameters
        ----------
        ma : Tensor (N, C_in, H, W)  activation means
        Sa : Tensor (N, C_in, H, W)  activation variances

        Returns
        -------
        mz : Tensor (N, C_out, H_out, W_out)  pre-activation means
        Sz : Tensor (N, C_out, H_out, W_out)  pre-activation variances
        """
        N, C, H, W = ma.shape
        self.input_shape = (N, C, H, W)

        # padding_type=2: right-bottom only (cuTAGI convention for stride>1 convs).
        # Pre-pad so im2col sees a (H+pad, W+pad) input with symmetric padding=0.
        if self.padding_type == 2 and self.padding > 0:
            p = self.padding
            ma_in = torch.nn.functional.pad(ma, (0, p, 0, p))
            Sa_in = torch.nn.functional.pad(Sa, (0, p, 0, p))
            H_out = (H + p - self.kH) // self.stride + 1
            W_out = (W + p - self.kW) // self.stride + 1
            im2col_pad = 0
        else:
            ma_in, Sa_in = ma, Sa
            H_out = (H + 2 * self.padding - self.kH) // self.stride + 1
            W_out = (W + 2 * self.padding - self.kW) // self.stride + 1
            im2col_pad = self.padding

        self.spatial = (H_out, W_out)

        # im2col: (N, C_in, H_in, W_in) → (N·L, K)
        patches_ma = _triton_im2col(ma_in, self.kH, self.kW, self.stride, im2col_pad)
        patches_Sa = _triton_im2col(Sa_in, self.kH, self.kW, self.stride, im2col_pad)
        self.patches_ma = patches_ma

        # Mean: cuBLAS matmul + bias
        mz_flat = torch.matmul(patches_ma, self.mw) + self.mb

        # Variance: Triton fused
        Sz_flat = triton_fused_var_forward(patches_ma, patches_Sa, self.mw, self.Sw, self.Sb)

        # Reshape (N·L, C_out) → (N, C_out, H_out, W_out)
        mz = mz_flat.view(N, H_out, W_out, self.C_out).permute(0, 3, 1, 2).contiguous()
        Sz = Sz_flat.view(N, H_out, W_out, self.C_out).permute(0, 3, 1, 2).contiguous()
        return mz, Sz

    # ------------------------------------------------------------------
    #  Backward (compute deltas only — NO parameter update)
    # ------------------------------------------------------------------
    def backward(self, delta_mz: Tensor, delta_Sz: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute parameter deltas and back-propagate innovation deltas.

        In patch space (P is the im2col patch matrix, shape N·L × K):

            Δμ_w = S_w · (P_μ^T @ δμ_z)       Δμ_b = S_b · Σ δμ_z
            ΔS_w = S_w² · (P_μ²)^T @ δS_z)    ΔS_b = S_b² · Σ δS_z

            δP_μ = δμ_z @ μ_w^T
            δP_S = δS_z @ (μ_w²)^T

        col2im folds (δP_μ, δP_S) back from patch space to spatial layout.

        Parameters
        ----------
        delta_mz : Tensor (N, C_out, H_out, W_out)
        delta_Sz : Tensor (N, C_out, H_out, W_out)

        Returns
        -------
        d_ma : Tensor (N, C_in, H, W)
        d_Sa : Tensor (N, C_in, H, W)
        """
        N = delta_mz.shape[0]

        # Flatten (N, C_out, H, W) → (N·L, C_out)
        dmz = delta_mz.permute(0, 2, 3, 1).reshape(-1, self.C_out).contiguous()
        dSz = delta_Sz.permute(0, 2, 3, 1).reshape(-1, self.C_out).contiguous()

        # ── Raw gradients (sum over all patches) ──
        grad_mw = torch.matmul(self.patches_ma.T, dmz)
        grad_mb = dmz.sum(0, keepdim=True)
        grad_Sw = torch.matmul((self.patches_ma**2).T, dSz)
        grad_Sb = dSz.sum(0, keepdim=True)

        # ── Parameter deltas (cuTAGI convention) ──
        self.delta_mw = self.Sw * grad_mw
        self.delta_Sw = (self.Sw**2) * grad_Sw
        self.delta_mb = self.Sb * grad_mb
        self.delta_Sb = (self.Sb**2) * grad_Sb

        # ── Delta propagation: Triton fused ──
        dp_ma, dp_Sa = triton_fused_backward_delta(dmz, dSz, self.mw)

        # ── col2im: (N·L, K) → (N, C_in, H, W) ──
        _, C, H, W = self.input_shape
        if self.padding_type == 2 and self.padding > 0:
            p = self.padding
            # col2im onto padded dims, then crop right/bottom padding away
            d_ma_p = _triton_col2im(dp_ma, N, C, H + p, W + p, self.kH, self.kW, self.stride, 0)
            d_Sa_p = _triton_col2im(dp_Sa, N, C, H + p, W + p, self.kH, self.kW, self.stride, 0)
            d_ma = d_ma_p[:, :, :H, :W].contiguous()
            d_Sa = d_Sa_p[:, :, :H, :W].contiguous()
        else:
            d_ma = _triton_col2im(dp_ma, N, C, H, W, self.kH, self.kW, self.stride, self.padding)
            d_Sa = _triton_col2im(dp_Sa, N, C, H, W, self.kH, self.kW, self.stride, self.padding)
        return d_ma, d_Sa

    # ------------------------------------------------------------------
    #  Update (apply capped deltas — called by the network)
    # ------------------------------------------------------------------
    def update(self, cap_factor: float) -> None:
        """Apply stored parameter deltas with cuTAGI-style capping."""
        update_parameters(self.mw, self.Sw, self.delta_mw, self.delta_Sw, cap_factor)
        update_parameters(self.mb, self.Sb, self.delta_mb, self.delta_Sb, cap_factor)

    @property
    def num_parameters(self) -> int:
        """Total learnable scalars: 2 × (weight + bias) means and variances."""
        return 2 * (self.mw.numel() + self.mb.numel())

    def __repr__(self):
        return (
            f"Conv2D({self.C_in}, {self.C_out}, kernel={self.kH}, "
            f"stride={self.stride}, pad={self.padding})"
        )
