"""
The autocov example network from docs/how_autocov_works.md, as a trainable Module:

    x ──▶ relu(x) ──▶ (relu(x) + x) ──▶ linear ──▶ y

Builds the graph, prints it, then runs ONE sample through a full TAGI step
(`observe`) and analyses what happened: the predictive distribution before the
update, the innovation, the output posterior, how the parameters move, and the
improved prediction afterwards. No dataset needed.

Usage:
    python examples/autocov_residual_graph.py
    python examples/autocov_residual_graph.py --compact
"""

from __future__ import annotations

import argparse
import math

import torch

from triton_tagi.autocov import Linear, Module, relu, tensor


class ResidualNet(Module):
    """x → relu(x) → (relu(x) + x) → linear → y."""

    def __init__(self, n_in: int, n_out: int, rng=None, device: str = "cpu"):
        super().__init__()
        self.linear = Linear(n_in, n_out, rng=rng, device=device)  # only learnable layer

    def forward(self, x):
        a = relu(x)            # Activation op
        b = a + x              # Add op — the residual "+ x" skip
        return self.linear(b)  # Linear op


def main(compact: bool = False) -> None:
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(0)

    net = ResidualNet(4, 1, rng=rng)
    net.assign_names()
    print(f"ResidualNet — parameters: {net.num_parameters}\n")

    # ── One sample ──
    x_sample = torch.tensor([[1.0, -0.5, 2.0, 0.3]])   # 1 sample, 4 features
    target = torch.tensor([[1.5]])                     # observed y
    var_v = 0.1                                        # observation noise variance

    # ── Forward: build the graph ──
    x = tensor(x_sample, var=0.0, name="x")
    y = net(x)
    print("Backward graph for y:")
    y.print_graph(compact=compact)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build, observe, and analyse one sample")
    p.add_argument("--compact", action="store_true", help="Flat numbered graph view")
    args = p.parse_args()
    main(compact=args.compact)
