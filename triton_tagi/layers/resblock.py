"""
TAGI-compatible Residual Block — exact replica of cuTAGI's ResNetBlock.

=======================================================================
  cuTAGI ResNetBlock Logic  (resnet_block.cpp / resnet_block_cuda.cu)
=======================================================================

Forward:
    1. Save a copy of the input (mu, var) for the shortcut path.
    2. Main path: Conv→ReLU→BN→Conv→ReLU→BN   (6 sub-layers)
    3. If projection shortcut:
           shortcut path: Conv(k=2,s=2)→ReLU→BN  on saved input
           output.mu += shortcut.mu ;  output.var += shortcut.var
       Else (identity):
           output.mu += saved_input.mu ;  output.var += saved_input.var
    4. NO activation after the addition.

Backward:
    1. Save a copy of the incoming deltas.
    2. Main path backward (BN→ReLU→Conv→BN→ReLU→Conv, reversed).
    3. If projection shortcut:
           shortcut backward on saved deltas
           output_delta += shortcut_delta   (simple sum)
       Else (identity):
           output_delta += saved_delta      (simple sum — no jcb scaling
           needed because our layer-by-layer backward already handles
           Jacobians; cuTAGI uses jcb scaling only because its layers
           embed the previous layer's Jacobian into their backward.)

=======================================================================
  Architecture Details
=======================================================================

  Main path:  Conv2D(3×3, stride) → ReLU → BN → Conv2D(3×3) → ReLU → BN
  Shortcut:   Identity   OR   Conv2D(2×2, stride=2) → ReLU → BN
  Merge:      element-wise addition of moments (no post-activation)

  A projection shortcut is used when stride > 1 or in_ch ≠ out_ch.
  cuTAGI uses kernel_size=2 for the projection conv (NOT 1×1).
"""

from __future__ import annotations

from torch import Tensor

from ..base import LearnableLayer
from .batchnorm2d import BatchNorm2D
from .conv2d import Conv2D
from .relu import ReLU


# ======================================================================
#  Shortcut / delta merge helpers (pure PyTorch, in-place)
#
#  Replicates cuTAGI's add_shortcut_mean_var_cuda: under the diagonal
#  independence approximation the cross-covariance is zero, so the shortcut
#  (or identity) moments simply add element-wise onto the main-path output.
#  The same element-wise sum merges the main- and shortcut-path deltas in the
#  backward pass.
# ======================================================================


def triton_add_shortcut(mu_s, var_s, mu_a, var_a):
    """
    In-place addition: mu_a += mu_s, var_a += var_s.
    Matches cuTAGI's add_shortcut_mean_var_cuda.
    """
    assert mu_s.shape == mu_a.shape, f"Shape mismatch: shortcut={mu_s.shape} vs output={mu_a.shape}"
    mu_a.add_(mu_s)
    var_a.add_(var_s)
    # mu_a, var_a modified in place


def triton_delta_merge(d_mu_skip, d_var_skip, d_mu_out, d_var_out):
    """
    In-place delta merge: d_mu_out += d_mu_skip, d_var_out += d_var_skip.
    Matches cuTAGI's backward add_shortcut_mean_var_cuda.
    """
    assert d_mu_skip.shape == d_mu_out.shape, (
        f"Shape mismatch: skip={d_mu_skip.shape} vs out={d_mu_out.shape}"
    )
    d_mu_out.add_(d_mu_skip)
    d_var_out.add_(d_var_skip)
    # d_mu_out, d_var_out modified in place


# ======================================================================
#  Add Layer — kept for backward compatibility / standalone use
# ======================================================================


class Add:
    """
    TAGI-compatible element-wise addition of two Gaussian streams.

    Forward:   μ_S = μ_Z + μ_X,   Σ_S = Σ_Z + Σ_X
    Backward:  deltas duplicated to both branches (Jacobian = 1).
    """

    def __init__(self) -> None:
        pass

    def forward(
        self, mu_z: Tensor, var_z: Tensor, mu_x: Tensor, var_x: Tensor
    ) -> tuple[Tensor, Tensor]:
        assert mu_z.shape == mu_x.shape
        mu_s = mu_z.clone()
        var_s = var_z.clone()
        triton_add_shortcut(mu_x, var_x, mu_s, var_s)
        return mu_s, var_s

    def backward(
        self, delta_mu_s: Tensor, delta_var_s: Tensor
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        d_mu_z = delta_mu_s.clone()
        d_var_z = delta_var_s.clone()
        d_mu_x = delta_mu_s.clone()
        d_var_x = delta_var_s.clone()
        return (d_mu_z, d_var_z), (d_mu_x, d_var_x)

    def __repr__(self):
        return "Add()"


# ======================================================================
#  ResBlock — exact replica of cuTAGI's ResNetBlock
# ======================================================================


class ResBlock(LearnableLayer):
    """
    TAGI Residual Block — replicates cuTAGI's ResNetBlock logic exactly.

    Architecture (from cuTAGI test_utils.cpp create_layer_block):
    ─────────────────────────────────────────────────────────────
    Main path:
        Conv2D(in_ch, out_ch, 3×3, stride, pad=1)
        → ReLU
        → BatchNorm2D(out_ch)
        → Conv2D(out_ch, out_ch, 3×3, stride=1, pad=1)
        → ReLU
        → BatchNorm2D(out_ch)

    Shortcut path (projection, when stride>1 or ch mismatch):
        Conv2D(in_ch, out_ch, 2×2, stride=2, pad=0)
        → ReLU
        → BatchNorm2D(out_ch)

    Shortcut path (identity, otherwise):
        pass-through

    Merge:
        output = main_output + shortcut_output     (no post-activation)

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    stride       : int  (default 1)
    device       : str  (default "cpu")
    gain_w, gain_b : float  (default 1.0)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        device: str = "cpu",
        gain_w: float = 1.0,
        gain_b: float = 1.0,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.device = device
        self.training = True

        # ── Main path: Conv→ReLU→BN→Conv→ReLU→BN ──
        # stride>1 blocks use padding_type=2 (right-bottom only) matching cuTAGI.
        self.conv1 = Conv2D(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            padding_type=2 if stride > 1 else 1,
            device=device,
            gain_w=gain_w,
            gain_b=gain_b,
        )
        self.relu1 = ReLU()
        self.bn1 = BatchNorm2D(
            out_channels, device=device, gain_w=gain_w, gain_b=gain_b, preserve_var=False
        )

        self.conv2 = Conv2D(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            device=device,
            gain_w=gain_w,
            gain_b=gain_b,
        )
        self.relu2 = ReLU()
        self.bn2 = BatchNorm2D(
            out_channels, device=device, gain_w=gain_w, gain_b=gain_b, preserve_var=False
        )

        # Ordered sub-layer list for main path
        self._main_layers = [self.conv1, self.relu1, self.bn1, self.conv2, self.relu2, self.bn2]

        # ── Shortcut path ──
        self.use_projection = (stride != 1) or (in_channels != out_channels)
        if self.use_projection:
            # cuTAGI uses kernel_size=2, stride=2, no bias, then ReLU→BN
            self.proj_conv = Conv2D(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=stride,
                padding=0,
                device=device,
                gain_w=gain_w,
                gain_b=gain_b,
            )
            self.proj_relu = ReLU()
            self.proj_bn = BatchNorm2D(
                out_channels, device=device, gain_w=gain_w, gain_b=gain_b, preserve_var=False
            )
            self._proj_layers = [self.proj_conv, self.proj_relu, self.proj_bn]
        else:
            self.proj_conv = None
            self.proj_relu = None
            self.proj_bn = None
            self._proj_layers = []

        # ── All learnable sub-layers (Conv2D, BatchNorm2D) ──
        self._learnable = [self.conv1, self.bn1, self.conv2, self.bn2]
        if self.use_projection:
            self._learnable.extend([self.proj_conv, self.proj_bn])

    # ------------------------------------------------------------------
    #  Train / Eval
    # ------------------------------------------------------------------
    def train(self) -> None:
        self.training = True
        for layer in self._learnable:
            if hasattr(layer, "train"):
                layer.train()

    def eval(self) -> None:
        self.training = False
        for layer in self._learnable:
            if hasattr(layer, "eval"):
                layer.eval()

    # ------------------------------------------------------------------
    #  Forward — replicates ResNetBlockCuda::forward exactly
    # ------------------------------------------------------------------
    def forward(self, mu_in: Tensor, var_in: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass through the residual block.

        cuTAGI logic:
            1. Save copy of input for shortcut.
            2. Main path forward.
            3. Add shortcut (projection or identity) to main output.
            4. No post-activation.
        """
        # Save input for shortcut path (like cuTAGI's input_z->copy_from)
        mu_skip = mu_in.clone()
        var_skip = var_in.clone()

        # ── Main path: Conv→ReLU→BN→Conv→ReLU→BN ──
        mu_z, var_z = mu_in, var_in
        for layer in self._main_layers:
            mu_z, var_z = layer.forward(mu_z, var_z)

        # ── Shortcut path ──
        if self.use_projection:
            mu_x, var_x = mu_skip, var_skip
            for layer in self._proj_layers:
                mu_x, var_x = layer.forward(mu_x, var_x)
        else:
            mu_x, var_x = mu_skip, var_skip

        # ── Merge: output += shortcut (in-place, like cuTAGI) ──
        triton_add_shortcut(mu_x, var_x, mu_z, var_z)

        # No activation after the addition (cuTAGI has none)
        return mu_z, var_z

    # ------------------------------------------------------------------
    #  Backward — replicates ResNetBlockCuda::backward exactly
    # ------------------------------------------------------------------
    def backward(self, delta_mu: Tensor, delta_var: Tensor) -> tuple[Tensor, Tensor]:
        """
        Backward pass through the residual block.

        cuTAGI logic:
            1. Save copy of incoming deltas for shortcut backward.
            2. Main path backward.
            3. If projection:  shortcut backward → add to main deltas.
               If identity:    add saved deltas directly to main deltas.
            4. Return merged deltas.

        Note: cuTAGI's identity backward uses jcb scaling:
            delta_mu_out += delta_mu_saved * jcb_input
            delta_var_out += delta_var_saved * jcb_input²
        In our framework, each layer's backward already applies its own
        Jacobian, so the identity shortcut is a simple addition (jcb=1.0
        effectively, since the reset-to-1 after the add means these deltas
        start with jcb=1.0).
        """
        # ── Split incoming deltas to both branches (no scaling) ──
        # The forward is a plain addition: out = main + skip
        # Both branches receive the full incoming delta (Jacobian = 1).
        d_mu_main = delta_mu.clone()
        d_var_main = delta_var.clone()

        d_mu_skip = delta_mu.clone()
        d_var_skip = delta_var.clone()

        # ── Main path backward (reversed) ──
        for layer in reversed(self._main_layers):
            d_mu_main, d_var_main = layer.backward(d_mu_main, d_var_main)

        # ── Shortcut path backward ──
        if self.use_projection:
            for layer in reversed(self._proj_layers):
                d_mu_skip, d_var_skip = layer.backward(d_mu_skip, d_var_skip)

        # ── Delta merge: main_delta += shortcut_delta (in-place) ──
        triton_delta_merge(d_mu_skip, d_var_skip, d_mu_main, d_var_main)

        return d_mu_main, d_var_main

    # ------------------------------------------------------------------
    #  Update
    # ------------------------------------------------------------------
    def update(self, cap_factor: float) -> None:
        """Apply capped parameter updates to all learnable sub-layers."""
        for layer in self._learnable:
            layer.update(cap_factor)

    # ------------------------------------------------------------------
    #  Properties for Sequential compatibility
    # ------------------------------------------------------------------
    @property
    def mw(self):
        return self.conv1.mw

    @mw.setter
    def mw(self, value):
        self.conv1.mw = value

    @property
    def Sw(self):
        return self.conv1.Sw

    @property
    def mb(self):
        return self.conv1.mb

    @property
    def Sb(self):
        return self.conv1.Sb

    @property
    def num_parameters(self) -> int:
        """Total learnable scalars across all sub-layers (means + variances)."""
        return sum(layer.num_parameters for layer in self._learnable)

    def __repr__(self):
        proj = "projection" if self.use_projection else "identity"
        return (
            f"ResBlock({self.in_channels}→{self.out_channels}, stride={self.stride}, skip={proj})"
        )
