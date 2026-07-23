"""
Build the autocov ResNet-18 and plot its backward graph — no data, no training.

Quick standalone script: constructs the network with the :mod:`triton_tagi.autocov`
graph engine, runs one dummy forward to build the graph, and prints the ASCII
backward graph (top → bottom = backward flow). Handy for inspecting structure.

Usage:
    python examples/cifar10_resnet18_autocov_copy.py
    python examples/cifar10_resnet18_autocov_copy.py --save graph.txt
    python examples/cifar10_resnet18_autocov_copy.py --show_moments
"""

from __future__ import annotations

import argparse

import torch

from triton_tagi.autocov import (
    AvgPool2D,
    BatchNorm2D,
    Conv2D,
    Flatten,
    Linear,
    Module,
    add,
    relu,
    remax,
    tensor,
)


# ---------------------------------------------------------------------------
#  Network — autocov ResBlock and ResNet-18
# ---------------------------------------------------------------------------

class ResBlock(Module):
    """TAGI residual block built from autocov ops (matches triton_tagi.ResBlock).

    Main path:  Conv(3×3,s) → ReLU → BN → Conv(3×3) → ReLU → BN
    Shortcut:   identity, or (stride>1 / channel change) Conv(2×2,s) → ReLU → BN
    Merge:      out = main + shortcut     (autocov ``add``; no post-activation)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        gain_w: float = 0.1,
        gain_b: float = 0.1,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        kw = dict(gain_w=gain_w, gain_b=gain_b, device=device)

        # ── Main path ── (stride>1 uses cuTAGI's right-bottom padding_type=2)
        self.conv1 = Conv2D(
            in_ch, out_ch, 3, stride=stride, padding=1,
            padding_type=2 if stride > 1 else 1, **kw,
        )
        self.bn1 = BatchNorm2D(out_ch, preserve_var=False, **kw)
        self.conv2 = Conv2D(out_ch, out_ch, 3, stride=1, padding=1, **kw)
        self.bn2 = BatchNorm2D(out_ch, preserve_var=False, **kw)

        # ── Shortcut path ──
        self.use_proj = (stride != 1) or (in_ch != out_ch)
        if self.use_proj:
            self.proj_conv = Conv2D(in_ch, out_ch, 2, stride=stride, padding=0, **kw)
            self.proj_bn = BatchNorm2D(out_ch, preserve_var=False, **kw)

    def forward(self, x):
        # Main path
        z = self.bn1(relu(self.conv1(x)))
        z = self.bn2(relu(self.conv2(z)))
        # Shortcut path
        s = self.proj_bn(relu(self.proj_conv(x))) if self.use_proj else x
        # Merge (residual add; variances add under the diagonal approximation)
        return add(z, s)


class ResNet18(Module):
    """CIFAR-10 ResNet-18 assembled from autocov :class:`ResBlock`s."""

    def __init__(
        self,
        num_classes: int = 10,
        gain_w: float = 0.1,
        gain_b: float = 0.1,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        kw = dict(gain_w=gain_w, gain_b=gain_b, device=device)

        # Stem: 32×32
        self.stem_conv = Conv2D(3, 64, 3, stride=1, padding=1, **kw)
        self.stem_bn = BatchNorm2D(64, **kw)

        # 4 stages × 2 blocks
        self.b1a = ResBlock(64, 64, 1, **kw)
        self.b1b = ResBlock(64, 64, 1, **kw)
        self.b2a = ResBlock(64, 128, 2, **kw)
        self.b2b = ResBlock(128, 128, 1, **kw)
        self.b3a = ResBlock(128, 256, 2, **kw)
        self.b3b = ResBlock(256, 256, 1, **kw)
        self.b4a = ResBlock(256, 512, 2, **kw)
        self.b4b = ResBlock(512, 512, 1, **kw)
        self._blocks = [
            self.b1a, self.b1b, self.b2a, self.b2b,
            self.b3a, self.b3b, self.b4a, self.b4b,
        ]

        # Head
        self.pool = AvgPool2D(4)   # 4×4 → 1×1
        self.flat = Flatten()      # 512
        self.fc = Linear(512, num_classes, **kw)

    def forward(self, x):
        h = self.stem_bn(relu(self.stem_conv(x)))
        for block in self._blocks:
            h = block(h)
        h = self.fc(self.flat(self.pool(h)))
        return remax(h)   # Remax classification head (lognormal, cuTAGI parity)


# ---------------------------------------------------------------------------
#  Build the network and plot its graph (no data, no training)
# ---------------------------------------------------------------------------

def main(save: str | None = None, show_moments: bool = False, compact: bool = False) -> None:
    torch.manual_seed(0)

    # ── Build the network ──
    net = ResNet18(device="cpu")
    net.assign_names()  # hierarchical display names (b2a.conv1, ...); wiring unaffected
    print(f"ResNet18 (autocov) — parameters: {net.num_parameters:,}")

    # ── One dummy forward (batch of 1) to build the graph ──
    x = torch.randn(1, 3, 32, 32)
    out = net(tensor(x, var=0.0))
    print(f"output shape: {tuple(out.shape)}  |  graph nodes: {len(out.build_topo())}\n")

    # ── Plot the backward graph (use --compact for the flat list view) ──
    graph = out.render_graph(show_moments=show_moments, compact=compact)
    print(graph)
    if save:
        with open(save, "w") as f:
            f.write(graph + "\n")
        print(f"\nGraph written to {save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build autocov ResNet-18 and plot its graph")
    parser.add_argument("--save", type=str, default=None, help="Write the graph to a text file")
    parser.add_argument("--show_moments", action="store_true", help="Annotate nodes with mu/var")
    parser.add_argument("--compact", action="store_true",
                        help="Flat numbered list (backward order) instead of the nested tree")
    args = parser.parse_args()
    main(save=args.save, show_moments=args.show_moments, compact=args.compact)
