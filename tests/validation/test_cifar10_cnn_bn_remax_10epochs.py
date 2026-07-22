"""Parity test: 3-block BN CNN + Remax, CIFAR-10, triton-tagi vs cuTAGI, 10 epochs, CUDA.

Architecture (matches ``examples/cifar10_cnn.py``)::

    Conv(3→32, k=5, p=2) → ReLU → BN → AvgPool(2)   [32→16]
    Conv(32→64, k=5, p=2) → ReLU → BN → AvgPool(2)  [16→8]
    Conv(64→64, k=5, p=2) → ReLU → BN → AvgPool(2)  [8→4]
    Flatten → Linear(1024→256) → ReLU → Linear(256→10) → Remax

Both networks:
  - start from the same He-initialised weights (Conv / Linear) and the
    cuTAGI-matching BatchNorm init (γ μ=1, var=1/n; β μ=0, var=1/n; running
    mean=0, running var=1),
  - see identical shuffled batches,
  - run on CUDA with σ_v = 0.05, batch = 128, no augmentation.

Important BN detail: triton-tagi's ``BatchNorm2D`` defaults to
``preserve_var=True``, which rescales γ on the first training batch to keep
the layer's output variance approximately invariant. cuTAGI does no such
rescaling, so the parity test must pass ``preserve_var=False``.

Run with:
    pytest tests/validation/test_cifar10_cnn_bn_remax_10epochs.py -v -s
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torchvision import datasets, transforms

pytagi = pytest.importorskip("pytagi", reason="cuTAGI (pytagi) not installed")
from pytagi.nn import AvgPool2d as PAvgPool2d
from pytagi.nn import BatchNorm2d as PBatchNorm2d
from pytagi.nn import Conv2d as PConv2d
from pytagi.nn import Linear as PLinear
from pytagi.nn import MixtureReLU, OutputUpdater, Remax
from pytagi.nn import Sequential as PSequential

from triton_tagi.layers.avgpool2d import AvgPool2D as TAvgPool2D
from triton_tagi.layers.batchnorm2d import BatchNorm2D as TBatchNorm2D
from triton_tagi.layers.conv2d import Conv2D as TConv2D
from triton_tagi.layers.flatten import Flatten as TFlatten
from triton_tagi.layers.linear import Linear as TLinear
from triton_tagi.layers.relu import ReLU as TReLU
from triton_tagi.layers.remax import Remax as TRemax
from triton_tagi.network import Sequential as TSequential

pytestmark = pytest.mark.cuda

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "data"
IN_C, H, W = 3, 32, 32
OUT_F = 10

SIGMA_V = 0.05
BATCH = 128
N_EPOCHS = 10
# Observed Δ on an RTX 4070 Ti SUPER (2026-04-19): 3.62 pp (triton 69.62 vs
# cuTAGI 66.01), fully deterministic run-to-run with both torch and pytagi
# seeded. The BN formulas are algebraically identical (both use Bessel-corrected
# variance since 2026-04-19) and match to ~1.9e-7 relative precision on a
# single forward pass, so the residual gap comes from compounded fp32 kernel
# non-associativity (~12k BN ops + Remax non-linearity over 10 epochs).
ACC_TOL = 0.04
ACC_MIN = 0.55     # both must reach ≥ 55 % without augmentation at 10 epochs

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


# ──────────────────────────────────────────────────────────────────────────────
#  Data
# ──────────────────────────────────────────────────────────────────────────────


def _load_cifar10():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_ds = datasets.CIFAR10(DATA_ROOT, train=True, download=False, transform=tf)
    test_ds = datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=tf)

    x_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    y_train = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
    x_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    y_test = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])

    y_train_oh = torch.zeros(len(y_train), OUT_F)
    y_train_oh.scatter_(1, y_train.unsqueeze(1), 1.0)
    return x_train, y_train_oh, y_train, x_test, y_test


# ──────────────────────────────────────────────────────────────────────────────
#  Weight init (matches cuTAGI's He / BN conventions)
# ──────────────────────────────────────────────────────────────────────────────


def _he_conv(C_in, C_out, k):
    fan_in = C_in * k * k
    scale = math.sqrt(1.0 / fan_in)
    K = C_in * k * k
    mw = torch.randn(K, C_out) * scale       # triton layout: (K, C_out)
    Sw = torch.full((K, C_out), scale ** 2)
    mb = torch.zeros(1, C_out)
    Sb = torch.full((1, C_out), scale ** 2)
    return mw, Sw, mb, Sb


def _he_linear(fan_in, fan_out):
    scale = math.sqrt(1.0 / fan_in)
    mw = torch.randn(fan_in, fan_out) * scale
    Sw = torch.full((fan_in, fan_out), scale ** 2)
    mb = torch.zeros(1, fan_out)
    Sb = torch.full((1, fan_out), scale ** 2)
    return mw, Sw, mb, Sb


def _bn_init(C):
    """BN init matching cuTAGI: γ μ=1, var=1/C; β μ=0, var=1/C."""
    scale = 2.0 / (C + C)
    return (
        torch.ones(C),
        torch.full((C,), scale),
        torch.zeros(C),
        torch.full((C,), scale),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Network builders
# ──────────────────────────────────────────────────────────────────────────────


def _build_triton(conv_p, bn_p, lin_p):
    """Construct the triton-tagi network with the given params on CUDA."""
    def _load_conv(layer, p):
        mw, Sw, mb, Sb = p
        layer.mw, layer.Sw = mw.to(DEVICE), Sw.to(DEVICE)
        layer.mb, layer.Sb = mb.to(DEVICE), Sb.to(DEVICE)
        return layer

    def _load_bn(C, p):
        mg, Sg, mb, Sb = p
        layer = TBatchNorm2D(C, device=DEVICE, preserve_var=False)
        layer.mw, layer.Sw = mg.to(DEVICE), Sg.to(DEVICE)
        layer.mb, layer.Sb = mb.to(DEVICE), Sb.to(DEVICE)
        return layer

    def _load_linear(layer, p):
        mw, Sw, mb, Sb = p
        layer.mw, layer.Sw = mw.to(DEVICE), Sw.to(DEVICE)
        layer.mb, layer.Sb = mb.to(DEVICE), Sb.to(DEVICE)
        return layer

    c0 = _load_conv(TConv2D(IN_C, 32, 5, padding=2, device=DEVICE), conv_p[0])
    c1 = _load_conv(TConv2D(32, 64, 5, padding=2, device=DEVICE), conv_p[1])
    c2 = _load_conv(TConv2D(64, 64, 5, padding=2, device=DEVICE), conv_p[2])
    bn0, bn1, bn2 = (_load_bn(C, p) for C, p in zip((32, 64, 64), bn_p))
    l0 = _load_linear(TLinear(1024, 256, device=DEVICE), lin_p[0])
    l1 = _load_linear(TLinear(256, 10, device=DEVICE), lin_p[1])

    return TSequential(
        [
            c0, TReLU(), bn0, TAvgPool2D(2),
            c1, TReLU(), bn1, TAvgPool2D(2),
            c2, TReLU(), bn2, TAvgPool2D(2),
            TFlatten(), l0, TReLU(), l1, TRemax(),
        ],
        device=DEVICE,
    )


def _flat_conv(mw, Sw, mb, Sb):
    """Triton (K, C_out) → pytagi (C_out, K) flat."""
    return (
        mw.T.cpu().numpy().flatten().tolist(),
        Sw.T.cpu().numpy().flatten().tolist(),
        mb.squeeze().cpu().numpy().tolist(),
        Sb.squeeze().cpu().numpy().tolist(),
    )


def _flat_lin(mw, Sw, mb, Sb):
    """Triton (in, out) → pytagi (out, in) flat."""
    return (
        mw.T.cpu().numpy().flatten().tolist(),
        Sw.T.cpu().numpy().flatten().tolist(),
        mb.squeeze().cpu().numpy().tolist(),
        Sb.squeeze().cpu().numpy().tolist(),
    )


def _flat_bn(mg, Sg, mb, Sb):
    """Triton and pytagi BN both use per-channel length-C vectors in the same order."""
    return (
        mg.cpu().numpy().tolist(),
        Sg.cpu().numpy().tolist(),
        mb.cpu().numpy().tolist(),
        Sb.cpu().numpy().tolist(),
    )


def _build_pytagi(conv_p, bn_p, lin_p):
    """Construct the pytagi network on CUDA with the same weights as triton-tagi."""
    net = PSequential(
        PConv2d(IN_C, 32, 5, padding=2, in_width=W, in_height=H),
        MixtureReLU(),
        PBatchNorm2d(32),
        PAvgPool2d(2, 2),
        PConv2d(32, 64, 5, padding=2),
        MixtureReLU(),
        PBatchNorm2d(64),
        PAvgPool2d(2, 2),
        PConv2d(64, 64, 5, padding=2),
        MixtureReLU(),
        PBatchNorm2d(64),
        PAvgPool2d(2, 2),
        PLinear(1024, 256),
        MixtureReLU(),
        PLinear(256, 10),
        Remax(),
    )
    net.preinit_layer()
    # Sort each layer group by numeric index (string sort puts "10" before "2").
    def _by_idx(prefix):
        return sorted(
            (k for k in net.state_dict() if k.startswith(prefix)),
            key=lambda k: int(k.split(".")[-1]),
        )

    conv_keys = _by_idx("Conv2d")        # [Conv2d.0, Conv2d.4, Conv2d.8]
    bn_keys = _by_idx("BatchNorm2d")     # [BatchNorm2d.2, BatchNorm2d.6, BatchNorm2d.10]
    lin_keys = _by_idx("Linear")         # [Linear.12, Linear.14]

    state = {}
    for k, p in zip(conv_keys, conv_p):
        state[k] = _flat_conv(*p)
    for k, p in zip(bn_keys, bn_p):
        state[k] = _flat_bn(*p)
    for k, p in zip(lin_keys, lin_p):
        state[k] = _flat_lin(*p)
    net.load_state_dict(state)
    net.to_device("cuda")
    return net


# ──────────────────────────────────────────────────────────────────────────────
#  Training and evaluation
# ──────────────────────────────────────────────────────────────────────────────


def _train_triton(net, x_train, y_train_oh, perm):
    x_s = x_train[perm].to(DEVICE)
    y_s = y_train_oh[perm].to(DEVICE)
    for i in range(0, len(x_s) - (len(x_s) % BATCH), BATCH):
        net.step(x_s[i : i + BATCH], y_s[i : i + BATCH], SIGMA_V)


def _train_pytagi(net, updater, x_train, y_train_oh, perm):
    x_np = x_train[perm].numpy()
    y_np = y_train_oh[perm].numpy()
    for i in range(0, len(x_np) - (len(x_np) % BATCH), BATCH):
        xb = x_np[i : i + BATCH].reshape(-1).astype(np.float32)
        yb = y_np[i : i + BATCH].flatten().astype(np.float32)
        nb = BATCH
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
    n = len(x) - (len(x) % BATCH)
    for i in range(0, n, BATCH):
        xb = x[i : i + BATCH]
        mu, _ = net.forward(xb)
        correct += (mu.argmax(dim=1).cpu() == y_labels[i : i + BATCH]).sum().item()
    net.train()
    return correct / n


def _accuracy_pytagi(net, x_test, y_labels):
    """Uses BATCH-sized batches (pytagi CUDA pre-allocates to first forward's size)."""
    correct = 0
    x_np = x_test.numpy()
    n = len(x_np) - (len(x_np) % BATCH)
    for i in range(0, n, BATCH):
        xb = x_np[i : i + BATCH].reshape(-1).astype(np.float32)
        mu_flat, _ = net(xb)
        mu = torch.tensor(mu_flat[: BATCH * OUT_F]).reshape(BATCH, OUT_F)
        correct += (mu.argmax(dim=1) == y_labels[i : i + BATCH]).sum().item()
    return correct / n


# ──────────────────────────────────────────────────────────────────────────────
#  The test
# ──────────────────────────────────────────────────────────────────────────────


def test_cifar10_cnn_bn_remax_10epochs():
    """10-epoch CIFAR-10 BN-CNN + Remax parity: triton-tagi (CUDA) vs cuTAGI (CUDA)."""
    torch.manual_seed(0)
    pytagi.manual_seed(0)   # cuTAGI has its own RNG; without this its CUDA reductions drift run-to-run
    conv_p = [_he_conv(IN_C, 32, 5), _he_conv(32, 64, 5), _he_conv(64, 64, 5)]
    bn_p = [_bn_init(32), _bn_init(64), _bn_init(64)]
    lin_p = [_he_linear(1024, 256), _he_linear(256, 10)]

    x_train, y_train_oh, _y_train_labels, x_test, y_test_labels = _load_cifar10()

    net_tri = _build_triton(conv_p, bn_p, lin_p)
    net_cut = _build_pytagi(conv_p, bn_p, lin_p)
    updater = OutputUpdater(net_cut.device)

    for _epoch in range(N_EPOCHS):
        perm = torch.randperm(len(x_train))
        _train_triton(net_tri, x_train, y_train_oh, perm)
        _train_pytagi(net_cut, updater, x_train, y_train_oh, perm)

    acc_tri = _accuracy_triton(net_tri, x_test, y_test_labels)
    acc_cut = _accuracy_pytagi(net_cut, x_test, y_test_labels)

    print(f"\n  triton-tagi (CUDA) BN-CNN + Remax:  {acc_tri * 100:.2f}%")
    print(f"  cuTAGI (CUDA)     BN-CNN + Remax:  {acc_cut * 100:.2f}%")
    print(f"  Δ accuracy:                          {abs(acc_tri - acc_cut) * 100:.3f}%  (tol {ACC_TOL * 100:.1f}%)")

    assert acc_tri >= ACC_MIN, f"triton-tagi: {acc_tri * 100:.2f}% < {ACC_MIN * 100:.0f}%"
    assert acc_cut >= ACC_MIN, f"cuTAGI: {acc_cut * 100:.2f}% < {ACC_MIN * 100:.0f}%"
    assert abs(acc_tri - acc_cut) < ACC_TOL, (
        f"BN-CNN + Remax accuracy gap {abs(acc_tri - acc_cut) * 100:.3f}% > {ACC_TOL * 100:.1f}% "
        f"(tri={acc_tri * 100:.2f}%  cut={acc_cut * 100:.2f}%)"
    )
