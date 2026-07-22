"""
Custom activation layer example: ELU (Exponential Linear Unit).

Shows how to implement a new activation layer in triton-tagi from first
principles.  ELU is a piecewise function defined as:

    f(z) = z              for z > 0
    f(z) = α(e^z − 1)    for z ≤ 0

Moment propagation (first-order Taylor for the nonlinear branch):

    For z > 0 (linear region):
        μ_a = μ_z
        J   = 1
        S_a = S_z

    For z ≤ 0 (exponential region, J = f'(μ_z) = α·e^μ_z):
        μ_a = α(e^μ_z − 1)
        J   = α · e^μ_z
        S_a = J² · S_z

Backward (same J stored from forward):

    δ_μ_z = J · δ_μ_a
    δ_S_z = J² · δ_S_a

Running this file trains a 3-layer MLP on MNIST for 5 epochs to demonstrate
that the ELU layer integrates correctly with Sequential and produces sensible
accuracy.

Usage:
    python examples/custom_layer.py
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from triton_tagi.base import Layer


# ======================================================================
#  Step 1 — the moment-propagation math, in pure PyTorch
#  Fuses forward + Jacobian computation in one vectorised pass.
# ======================================================================


def _elu_forward(mz: Tensor, Sz: Tensor, alpha: float):
    pos = mz > 0.0

    # Jacobian: 1 in the positive branch, α·e^μ_z in the negative branch
    J = torch.where(pos, torch.ones_like(mz), alpha * torch.exp(mz))

    # Mean: μ_z in the positive branch, α(e^μ_z − 1) in the negative branch
    ma = torch.where(pos, mz, alpha * (torch.exp(mz) - 1.0))

    # Variance: J² · S_z (guard against fp32 underflow)
    Sa = (J * J * Sz).clamp_min(0.0)

    return ma, Sa, J


def _elu_backward(J: Tensor, delta_ma: Tensor, delta_Sa: Tensor):
    return J * delta_ma, J * J * delta_Sa


# ======================================================================
#  Step 3 — Layer class
# ======================================================================


class ELU(Layer):
    """
    Bayesian ELU activation layer.

    Propagates Gaussian moments through f(z) = z for z > 0,
    α(e^z − 1) for z ≤ 0, using a first-order Taylor approximation
    in the exponential branch.

    Parameters
    ----------
    alpha : float  slope of the negative branch (default 1.0)
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._J: Tensor | None = None

    def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]:
        """
        Propagate Gaussian moments through ELU.

        μ_a = μ_z · 1_{μ_z>0}  +  α(e^μ_z − 1) · 1_{μ_z≤0}
        J   = 1_{μ_z>0}         +  α·e^μ_z · 1_{μ_z≤0}
        S_a = J² · S_z

        Parameters
        ----------
        ma : Tensor  pre-activation means, any shape
        Sa : Tensor  pre-activation variances, same shape as ma

        Returns
        -------
        ma_out : Tensor  post-activation means
        Sa_out : Tensor  post-activation variances
        """
        shape = ma.shape
        ma_out, Sa_out, self._J = _elu_forward(
            ma.reshape(-1), Sa.reshape(-1), self.alpha
        )
        return ma_out.reshape(shape), Sa_out.reshape(shape)

    def backward(self, delta_ma: Tensor, delta_Sa: Tensor) -> tuple[Tensor, Tensor]:
        """
        Back-propagate innovation deltas through ELU.

        δ_μ_z = J · δ_μ_a
        δ_S_z = J² · δ_S_a

        Parameters
        ----------
        delta_ma : Tensor  mean innovation deltas from the next layer
        delta_Sa : Tensor  variance innovation deltas from the next layer

        Returns
        -------
        d_ma : Tensor  mean deltas to propagate to the previous layer
        d_Sa : Tensor  variance deltas to propagate to the previous layer
        """
        shape = delta_ma.shape
        d_ma, d_Sa = _elu_backward(
            self._J,
            delta_ma.reshape(-1),
            delta_Sa.reshape(-1),
        )
        return d_ma.reshape(shape), d_Sa.reshape(shape)

    def __repr__(self) -> str:
        return f"ELU(alpha={self.alpha})"


# ======================================================================
#  Step 4 — Quick smoke test + MNIST demo
# ======================================================================


def _unit_check():
    """Verify shapes, non-negativity of Sa, and zero-delta passthrough."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = ELU(alpha=1.0)

    ma = torch.randn(8, 32, device=device)
    Sa = torch.rand(8, 32, device=device) * 0.1

    ma_out, Sa_out = layer.forward(ma, Sa)
    assert ma_out.shape == ma.shape, "forward shape mismatch"
    assert (Sa_out >= 0).all(), "Sa_out must be non-negative"

    # Zero-input → negative mean, non-zero variance must propagate
    ma_zero = torch.zeros(4, 16, device=device)
    Sa_zero = torch.zeros(4, 16, device=device)
    ma_o, Sa_o = layer.forward(ma_zero, Sa_zero)
    assert torch.allclose(Sa_o, Sa_zero), "Sa_out should be 0 when Sa=0"

    dma = torch.zeros_like(ma_out)
    dSa = torch.zeros_like(Sa_out)
    d_ma, d_Sa = layer.backward(dma, dSa)
    assert torch.allclose(d_ma, torch.zeros_like(d_ma)), "zero delta passthrough"

    print("Unit checks passed.")


def _mnist_demo():
    """Train a 3-layer MLP with ELU activations on MNIST for 5 epochs."""
    import os

    import numpy as np
    from torchvision import datasets, transforms

    from triton_tagi.layers import Flatten, Linear
    from triton_tagi.network import Sequential
    from triton_tagi.update.observation import compute_innovation
    from triton_tagi.update.parameters import get_cap_factor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    B, SIGMA_V = 256, 0.1
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=B, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=B, shuffle=False)

    net = Sequential(
        [
            Flatten(),
            Linear(784, 256, device=device),
            ELU(),
            Linear(256, 128, device=device),
            ELU(),
            Linear(128, 10, device=device),
        ],
        device=device,
    )
    net.train()
    print(f"Parameters: {net.num_parameters()}")

    for epoch in range(1, 6):
        net.train()
        for x, y in train_loader:
            x = x.to(device)
            ma = x.reshape(x.size(0), -1)
            Sa = torch.zeros_like(ma)
            for layer in net.layers:
                ma, Sa = layer.forward(ma, Sa)

            y_oh = torch.zeros(x.size(0), 10, device=device)
            y_oh.scatter_(1, y.to(device).unsqueeze(1), 1.0)
            delta_ma, delta_Sa = compute_innovation(y_oh, ma, Sa, SIGMA_V)

            cap = get_cap_factor(x.size(0))
            for layer in reversed(net.layers):
                delta_ma, delta_Sa = layer.backward(delta_ma, delta_Sa)
            for layer in net.layers:
                if hasattr(layer, "update"):
                    layer.update(cap)

        # Eval
        net.eval()
        correct = total = 0
        for x, y in test_loader:
            x = x.to(device)
            ma = x.reshape(x.size(0), -1)
            Sa = torch.zeros_like(ma)
            for layer in net.layers:
                ma, Sa = layer.forward(ma, Sa)
            pred = ma.argmax(dim=1).cpu()
            correct += (pred == y).sum().item()
            total += y.size(0)
        print(f"Epoch {epoch}: test accuracy = {100*correct/total:.2f}%")


if __name__ == "__main__":
    _unit_check()
    _mnist_demo()
