"""Validation test: triton-tagi MLP vs cuTAGI (pytagi) on MNIST.

Both networks start from identical weights, see identical batches in identical
order, and use identical hyperparameters.  After 5 epochs both must reach
≥ 93 % accuracy and be within 3 percentage points of each other.

Architecture:  784 → 256 → 128 → 10  (Linear–ReLU–Linear–ReLU–Linear)
Optimizer:     standard TAGI (no Adam/momentum)
σ_v:           0.05
Batch size:    512
Epochs:        5
Label encoding: one-hot {0, 1}  (class = 1, rest = 0)

Tolerance rationale
-------------------
Both implementations use identical exact Gaussian moment formulas for ReLU
(cuTAGI mixture_relu_mean_var and triton-tagi bayesian_relu are algebraically
the same).  Any residual gap is purely a fp32 arithmetic artefact: pytagi runs
on CPU (C++ FMA/SIMD), triton-tagi runs on GPU (Triton tiles + cuBLAS).  The
same formula rounds differently depending on hardware and operation order, and
these per-operation differences accumulate over ~585 training steps into a
small accuracy gap.  The 1 % tolerance is set above this expected fp32 noise
so that real bugs (wrong formula, wrong sigma_v, sign errors) still fail while
normal hardware-induced accumulation passes.

Timing: ~220 s on a single GPU (pytagi is the bottleneck, ~30 s/epoch).

Run with:
    pytest tests/validation/test_mnist_mlp.py -v -s
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from pytagi.nn import Linear as PLinear
from pytagi.nn import MixtureReLU, OutputUpdater
from pytagi.nn import Sequential as PSequential
from torchvision import datasets

from triton_tagi.layers.linear import Linear as TLinear
from triton_tagi.layers.relu import ReLU as TReLU
from triton_tagi.network import Sequential as TSequential

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "data"

pytestmark = pytest.mark.cuda

# ── Hyperparameters ──
IN_F, H1, H2, OUT_F = 784, 256, 128, 10
BATCH = 512
SIGMA_V = 0.05
N_EPOCHS = 5
ACC_TOL = 0.01    # 1 percentage point (see docstring for rationale)
ACC_MIN = 0.93    # both implementations must reach ≥ 93 %


# ──────────────────────────────────────────────────────────────────────────────
#  Data helpers
# ──────────────────────────────────────────────────────────────────────────────


def _load_mnist():
    """Load and normalise MNIST; return (x_train, y_train_oh, y_train_labels,
    x_test, y_test_labels) as float32 CPU tensors."""
    train_ds = datasets.MNIST(DATA_ROOT, train=True, download=False)
    test_ds = datasets.MNIST(DATA_ROOT, train=False, download=False)

    x_train = train_ds.data.float().view(-1, 784) / 255.0
    x_test = test_ds.data.float().view(-1, 784) / 255.0
    mu, sigma = x_train.mean(), x_train.std()
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma

    y_labels_train = train_ds.targets  # (60000,)  LongTensor
    y_labels_test = test_ds.targets    # (10000,)

    # One-hot {0, 1}
    y_train_oh = torch.zeros(len(y_labels_train), OUT_F)
    y_train_oh.scatter_(1, y_labels_train.unsqueeze(1), 1.0)

    return x_train, y_train_oh, y_labels_train, x_test, y_labels_test


def _he_init(fan_in: int, fan_out: int):
    """He init matching cuTAGI: scale = sqrt(1/fan_in), Sw = 1/fan_in."""
    scale = math.sqrt(1.0 / fan_in)
    mw = torch.randn(fan_in, fan_out) * scale
    Sw = torch.full((fan_in, fan_out), scale ** 2)
    mb = torch.randn(1, fan_out) * scale
    Sb = torch.full((1, fan_out), scale ** 2)
    return mw, Sw, mb, Sb


# ──────────────────────────────────────────────────────────────────────────────
#  Network builders
# ──────────────────────────────────────────────────────────────────────────────


def _build_triton(params):
    """Build triton-tagi 3-layer MLP with the given parameters."""
    (mw0, Sw0, mb0, Sb0), (mw1, Sw1, mb1, Sb1), (mw2, Sw2, mb2, Sb2) = params
    l0 = TLinear(IN_F, H1, device=DEVICE)
    l0.mw, l0.Sw, l0.mb, l0.Sb = mw0.to(DEVICE), Sw0.to(DEVICE), mb0.to(DEVICE), Sb0.to(DEVICE)
    l1 = TLinear(H1, H2, device=DEVICE)
    l1.mw, l1.Sw, l1.mb, l1.Sb = mw1.to(DEVICE), Sw1.to(DEVICE), mb1.to(DEVICE), Sb1.to(DEVICE)
    l2 = TLinear(H2, OUT_F, device=DEVICE)
    l2.mw, l2.Sw, l2.mb, l2.Sb = mw2.to(DEVICE), Sw2.to(DEVICE), mb2.to(DEVICE), Sb2.to(DEVICE)
    return TSequential([l0, TReLU(), l1, TReLU(), l2], device=DEVICE)


def _build_pytagi(params):
    """Build pytagi 3-layer MLP with identical weights."""
    (mw0, Sw0, mb0, Sb0), (mw1, Sw1, mb1, Sb1), (mw2, Sw2, mb2, Sb2) = params

    def _flat(mw, Sw, mb, Sb):
        return (
            mw.T.numpy().flatten().tolist(),
            Sw.T.numpy().flatten().tolist(),
            mb.squeeze().numpy().tolist(),
            Sb.squeeze().numpy().tolist(),
        )

    net = PSequential(PLinear(IN_F, H1), MixtureReLU(), PLinear(H1, H2), MixtureReLU(), PLinear(H2, OUT_F))
    net.preinit_layer()
    keys = sorted(net.state_dict().keys())   # ["Linear.0", "Linear.2", "Linear.4"]
    net.load_state_dict({
        keys[0]: _flat(mw0, Sw0, mb0, Sb0),
        keys[1]: _flat(mw1, Sw1, mb1, Sb1),
        keys[2]: _flat(mw2, Sw2, mb2, Sb2),
    })
    return net


# ──────────────────────────────────────────────────────────────────────────────
#  Training & evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────


def _train_epoch_triton(net, x_train, y_train_oh, perm):
    """One epoch of triton-tagi training."""
    x_s = x_train[perm].to(DEVICE)
    y_s = y_train_oh[perm].to(DEVICE)
    for i in range(0, len(x_s), BATCH):
        xb = x_s[i : i + BATCH]
        yb = y_s[i : i + BATCH]
        net.step(xb, yb, SIGMA_V)


def _train_epoch_pytagi(net, updater, x_train, y_train_oh, perm):
    """One epoch of pytagi training."""
    x_np = x_train[perm].numpy()
    y_np = y_train_oh[perm].numpy()
    for i in range(0, len(x_np), BATCH):
        xb = x_np[i : i + BATCH].flatten().astype(np.float32)
        yb = y_np[i : i + BATCH].flatten().astype(np.float32)
        nb = len(yb) // OUT_F
        var_yb = np.full(nb * OUT_F, SIGMA_V ** 2, dtype=np.float32)
        net(xb)
        updater.update(
            output_states=net.output_z_buffer,
            mu_obs=yb,
            var_obs=var_yb,
            delta_states=net.input_delta_z_buffer,
        )
        net.backward()
        net.step()


def _accuracy_triton(net, x_test, y_labels):
    correct = 0
    x = x_test.to(DEVICE)
    for i in range(0, len(x), 1024):
        xb = x[i : i + 1024]
        mu, _ = net.forward(xb)
        correct += (mu.argmax(dim=1).cpu() == y_labels[i : i + 1024]).sum().item()
    return correct / len(y_labels)


def _accuracy_pytagi(net, x_test, y_labels):
    correct = 0
    x_np = x_test.numpy()
    for i in range(0, len(x_np), 1024):
        xb = x_np[i : i + 1024]
        nb = len(xb)
        mu_flat, _ = net(xb.flatten().astype(np.float32))
        mu = torch.tensor(mu_flat).reshape(nb, OUT_F)
        correct += (mu.argmax(dim=1) == y_labels[i : i + 1024]).sum().item()
    return correct / len(y_labels)


# ──────────────────────────────────────────────────────────────────────────────
#  The test
# ──────────────────────────────────────────────────────────────────────────────


def test_mnist_mlp_5epochs():
    """After 5 epochs both implementations reach ≥ 93 % and are within 3 %."""
    # ── Shared initialisation ──
    torch.manual_seed(0)
    params = [_he_init(IN_F, H1), _he_init(H1, H2), _he_init(H2, OUT_F)]

    # ── Data ──
    x_train, y_train_oh, y_train_labels, x_test, y_test_labels = _load_mnist()

    # ── Build networks ──
    net_tri = _build_triton(params)
    net_cut = _build_pytagi(params)
    updater = OutputUpdater(net_cut.device)

    # ── Training ──
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(len(x_train))
        _train_epoch_triton(net_tri, x_train, y_train_oh, perm)
        _train_epoch_pytagi(net_cut, updater, x_train, y_train_oh, perm)

    # ── Accuracy ──
    acc_tri = _accuracy_triton(net_tri, x_test, y_test_labels)
    acc_cut = _accuracy_pytagi(net_cut, x_test, y_test_labels)

    print(f"\n  triton-tagi:  {acc_tri * 100:.2f}%")
    print(f"  cuTAGI:       {acc_cut * 100:.2f}%")
    print(f"  Δ accuracy:   {abs(acc_tri - acc_cut) * 100:.3f}%  (tol {ACC_TOL*100:.1f}%)")

    assert acc_tri >= ACC_MIN, (
        f"triton-tagi failed to converge: {acc_tri*100:.2f}% < {ACC_MIN*100:.0f}% floor"
    )
    assert acc_cut >= ACC_MIN, (
        f"cuTAGI failed to converge: {acc_cut*100:.2f}% < {ACC_MIN*100:.0f}% floor"
    )
    assert abs(acc_tri - acc_cut) < ACC_TOL, (
        f"Accuracy gap {abs(acc_tri - acc_cut)*100:.3f}% exceeds {ACC_TOL*100:.1f}% tolerance. "
        f"triton={acc_tri*100:.2f}%  cuTAGI={acc_cut*100:.2f}%"
    )
