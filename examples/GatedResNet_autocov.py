"""
GatedResNet (autocov) — build the network and plot its backward graph.

No data, no training: construct the gated residual net with the
:mod:`triton_tagi.autocov` graph engine, run one dummy forward, and print the
backward graph.

    a      = relu(inp(x))
    branch = tanh(h(a)) * sigmoid(gate(a))     # gated multiplicative branch
    out    = out(a + branch)                   # skip-add

Usage:
    python examples/GatedResNet_autocov.py
    python examples/GatedResNet_autocov.py --compact
"""

from __future__ import annotations

import argparse

import torch

from triton_tagi.autocov import Linear, Module, relu, sigmoid, tanh, tensor


class GatedResNet(Module):
    def __init__(self, rng=None, device="cpu"):
        super().__init__()
        self.inp = Linear(1, 32, rng=rng, device=device)
        self.h = Linear(32, 32, rng=rng, device=device)
        self.gate = Linear(32, 32, rng=rng, device=device)
        self.out = Linear(32, 1, rng=rng, device=device)

    def forward(self, x):
        a = relu(self.inp(x))
        branch = tanh(self.h(a)) * sigmoid(self.gate(a))  # gated multiplicative branch
        return self.out(a + branch)                       # skip add


def main(compact: bool = False, show_moments: bool = False) -> None:
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(0)

    net = GatedResNet(rng=rng)
    net.assign_names()  # hierarchical display names (inp, h, gate, out); wiring unaffected
    print(f"GatedResNet (autocov) — parameters: {net.num_parameters:,}")

    out = net(tensor(torch.randn(4, 1), var=0.0))
    print(f"output shape: {tuple(out.shape)}  |  graph nodes: {len(out.build_topo())}\n")

    out.print_graph(compact=compact, show_moments=show_moments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the autocov GatedResNet and plot its graph")
    parser.add_argument("--compact", action="store_true", help="Flat numbered list view")
    parser.add_argument("--show_moments", action="store_true", help="Annotate nodes with mu/var")
    args = parser.parse_args()
    main(compact=args.compact, show_moments=args.show_moments)
