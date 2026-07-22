"""Parity test: triton-tagi vs cuTAGI (pytagi), both on CUDA, MNIST + Remax, 10 epochs.

Both networks:
  - start from the same He-initialised weights,
  - see the same shuffled mini-batches in the same order,
  - run on CUDA,
  - end their forward pass with ``Remax`` (Bayesian ReLU-normalised softmax),
  - use identical σ_v = 0.05 and one-hot targets.

The assertion is that after 10 epochs both reach ≥ 96 % on the MNIST test set
and agree to within a small tolerance. Because both runs use CUDA fp32 but
different kernel implementations (cuTAGI C++/CUDA vs triton-tagi Triton +
cuBLAS), accumulation order differs per kernel and a small final-epoch gap is
expected. The tolerance is set above that hardware-induced drift so real bugs
(wrong formula, wrong σ_v, sign errors) still fail.

Run with:
    pytest tests/validation/test_mnist_remax_10epochs.py -v -s
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import pytagi
from pytagi.nn import Linear as PLinear
from pytagi.nn import MixtureReLU, OutputUpdater, Remax
from pytagi.nn import Sequential as PSequential
from torchvision import datasets

from triton_tagi.layers.linear import Linear as TLinear
from triton_tagi.layers.relu import ReLU as TReLU
from triton_tagi.layers.remax import Remax as TRemax
from triton_tagi.network import Sequential as TSequential

pytestmark = pytest.mark.cuda

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "data"

IN_F, H1, H2, OUT_F = 784, 256, 128, 10
BATCH = 512
SIGMA_V = 0.05
N_EPOCHS = 10
# Observed Δ on an RTX 4070 Ti SUPER (2026-04-19): 0.21 pp (triton 98.04 vs
# cuTAGI 97.83). Tolerance set to 0.5 pp — well above the measured noise,
# well below the ~1 pp you'd see with a real bug (wrong formula, wrong σ_v).
ACC_TOL = 0.005
ACC_MIN = 0.96      # both must reach ≥ 96 %


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _load_mnist():
    train_ds = datasets.MNIST(DATA_ROOT, train=True, download=False)
    test_ds = datasets.MNIST(DATA_ROOT, train=False, download=False)

    x_train = train_ds.data.float().view(-1, IN_F) / 255.0
    x_test = test_ds.data.float().view(-1, IN_F) / 255.0
    mu, sigma = x_train.mean(), x_train.std()
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma

    y_labels_train = train_ds.targets
    y_labels_test = test_ds.targets

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


def _build_triton(params):
    (mw0, Sw0, mb0, Sb0), (mw1, Sw1, mb1, Sb1), (mw2, Sw2, mb2, Sb2) = params
    l0 = TLinear(IN_F, H1, device=DEVICE)
    l0.mw, l0.Sw, l0.mb, l0.Sb = mw0.to(DEVICE), Sw0.to(DEVICE), mb0.to(DEVICE), Sb0.to(DEVICE)
    l1 = TLinear(H1, H2, device=DEVICE)
    l1.mw, l1.Sw, l1.mb, l1.Sb = mw1.to(DEVICE), Sw1.to(DEVICE), mb1.to(DEVICE), Sb1.to(DEVICE)
    l2 = TLinear(H2, OUT_F, device=DEVICE)
    l2.mw, l2.Sw, l2.mb, l2.Sb = mw2.to(DEVICE), Sw2.to(DEVICE), mb2.to(DEVICE), Sb2.to(DEVICE)
    return TSequential([l0, TReLU(), l1, TReLU(), l2, TRemax()], device=DEVICE)


def _build_pytagi(params):
    """Build pytagi MLP + Remax on CUDA with the same weights as triton-tagi.

    Weights must be loaded BEFORE `to_device('cuda')` because the state-dict
    keys change from ``Linear.N`` to ``LinearCuda.N`` after the device move.
    """
    (mw0, Sw0, mb0, Sb0), (mw1, Sw1, mb1, Sb1), (mw2, Sw2, mb2, Sb2) = params

    def _flat(mw, Sw, mb, Sb):
        return (
            mw.T.numpy().flatten().tolist(),
            Sw.T.numpy().flatten().tolist(),
            mb.squeeze().numpy().tolist(),
            Sb.squeeze().numpy().tolist(),
        )

    net = PSequential(
        PLinear(IN_F, H1), MixtureReLU(),
        PLinear(H1, H2), MixtureReLU(),
        PLinear(H2, OUT_F), Remax(),
    )
    net.preinit_layer()
    keys = sorted(net.state_dict().keys())       # Linear.0, Linear.2, Linear.4
    net.load_state_dict({
        keys[0]: _flat(mw0, Sw0, mb0, Sb0),
        keys[1]: _flat(mw1, Sw1, mb1, Sb1),
        keys[2]: _flat(mw2, Sw2, mb2, Sb2),
    })
    net.to_device("cuda")
    return net


def _train_triton(net, x_train, y_train_oh, perm):
    x_s = x_train[perm].to(DEVICE)
    y_s = y_train_oh[perm].to(DEVICE)
    for i in range(0, len(x_s), BATCH):
        net.step(x_s[i : i + BATCH], y_s[i : i + BATCH], SIGMA_V)


def _train_pytagi(net, updater, x_train, y_train_oh, perm):
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
    net.eval()
    correct = 0
    x = x_test.to(DEVICE)
    for i in range(0, len(x), BATCH):
        xb = x[i : i + BATCH]
        mu, _ = net.forward(xb)
        correct += (mu.argmax(dim=1).cpu() == y_labels[i : i + BATCH]).sum().item()
    net.train()
    return correct / len(y_labels)


def _accuracy_pytagi(net, x_test, y_labels):
    """Evaluate pytagi at BATCH size — pytagi CUDA pre-allocates buffers to the
    first forward's batch size and crashes on a larger batch, so eval must not
    exceed the training batch size.
    """
    correct = 0
    x_np = x_test.numpy()
    for i in range(0, len(x_np), BATCH):
        xb = x_np[i : i + BATCH]
        nb = len(xb)
        mu_flat, _ = net(xb.flatten().astype(np.float32))
        mu = torch.tensor(mu_flat[: nb * OUT_F]).reshape(nb, OUT_F)
        correct += (mu.argmax(dim=1) == y_labels[i : i + BATCH]).sum().item()
    return correct / len(y_labels)


# ──────────────────────────────────────────────────────────────────────────────
#  The test
# ──────────────────────────────────────────────────────────────────────────────


def test_mnist_remax_10epochs():
    """10-epoch MNIST Remax parity: triton-tagi (CUDA) vs cuTAGI (CUDA)."""
    torch.manual_seed(0)
    pytagi.manual_seed(0)   # make cuTAGI's CUDA reductions deterministic run-to-run
    params = [_he_init(IN_F, H1), _he_init(H1, H2), _he_init(H2, OUT_F)]

    x_train, y_train_oh, _y_train_labels, x_test, y_test_labels = _load_mnist()

    net_tri = _build_triton(params)
    net_cut = _build_pytagi(params)
    updater = OutputUpdater(net_cut.device)

    for _epoch in range(N_EPOCHS):
        perm = torch.randperm(len(x_train))
        _train_triton(net_tri, x_train, y_train_oh, perm)
        _train_pytagi(net_cut, updater, x_train, y_train_oh, perm)

    acc_tri = _accuracy_triton(net_tri, x_test, y_test_labels)
    acc_cut = _accuracy_pytagi(net_cut, x_test, y_test_labels)

    print(f"\n  triton-tagi (CUDA) + Remax:  {acc_tri * 100:.2f}%")
    print(f"  cuTAGI (CUDA)     + Remax:  {acc_cut * 100:.2f}%")
    print(f"  Δ accuracy:                  {abs(acc_tri - acc_cut) * 100:.3f}%  (tol {ACC_TOL * 100:.1f}%)")

    assert acc_tri >= ACC_MIN, f"triton-tagi Remax: {acc_tri * 100:.2f}% < {ACC_MIN * 100:.0f}%"
    assert acc_cut >= ACC_MIN, f"cuTAGI Remax: {acc_cut * 100:.2f}% < {ACC_MIN * 100:.0f}%"
    assert abs(acc_tri - acc_cut) < ACC_TOL, (
        f"Remax accuracy gap {abs(acc_tri - acc_cut) * 100:.3f}% > {ACC_TOL * 100:.1f}% "
        f"(tri={acc_tri * 100:.2f}%  cut={acc_cut * 100:.2f}%)"
    )
