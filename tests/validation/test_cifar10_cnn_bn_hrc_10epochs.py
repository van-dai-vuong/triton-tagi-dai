"""Parity test: 3-block BN CNN + HRC, CIFAR-10, triton-tagi vs cuTAGI, 10 epochs, CUDA.

Architecture (matches ``examples/cifar10_cnn.py`` with the Remax head replaced
by a hierarchical-softmax head)::

    Conv(3→32, k=5, p=2) → ReLU → BN → AvgPool(2)   [32→16]
    Conv(32→64, k=5, p=2) → ReLU → BN → AvgPool(2)  [16→8]
    Conv(64→64, k=5, p=2) → ReLU → BN → AvgPool(2)  [8→4]
    Flatten → Linear(1024→256) → ReLU → Linear(256→11)   (HRC tree for 10 classes)

Both networks:
  - start from the same He-initialised weights and cuTAGI-matching BN init,
  - see identical shuffled batches,
  - run on CUDA with σ_v = 0.05, batch = 128, no augmentation,
  - use hierarchical softmax output: 11 neurons = 4 observation indices per
    sample selecting a binary path through the class tree.

As in the Remax variant, triton-tagi's BN ``preserve_var=True`` default is
overridden to match cuTAGI (no first-batch γ rescaling).

Run with:
    pytest tests/validation/test_cifar10_cnn_bn_hrc_10epochs.py -v -s
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torchvision import datasets, transforms

pytagi = pytest.importorskip("pytagi", reason="cuTAGI (pytagi) not installed")
from pytagi import HRCSoftmaxMetric, Utils
from pytagi.nn import AvgPool2d as PAvgPool2d
from pytagi.nn import BatchNorm2d as PBatchNorm2d
from pytagi.nn import Conv2d as PConv2d
from pytagi.nn import Linear as PLinear
from pytagi.nn import MixtureReLU, OutputUpdater
from pytagi.nn import Sequential as PSequential

from triton_tagi.hrc_softmax import class_to_obs, get_predicted_labels
from triton_tagi.layers.avgpool2d import AvgPool2D as TAvgPool2D
from triton_tagi.layers.batchnorm2d import BatchNorm2D as TBatchNorm2D
from triton_tagi.layers.conv2d import Conv2D as TConv2D
from triton_tagi.layers.flatten import Flatten as TFlatten
from triton_tagi.layers.linear import Linear as TLinear
from triton_tagi.layers.relu import ReLU as TReLU
from triton_tagi.network import Sequential as TSequential

pytestmark = pytest.mark.cuda

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "data"
IN_C, H, W = 3, 32, 32
N_CLASSES = 10
HRC_LEN = 11       # class_to_obs(10).len

SIGMA_V = 0.05
BATCH = 128
N_EPOCHS = 10
# Observed Δ on an RTX 4070 Ti SUPER (2026-04-19): 0.55 pp (triton 71.36 vs
# cuTAGI 71.92), fully deterministic with torch and pytagi both seeded.
# HRC's sparse output update (4 indices per sample) exposes less of the
# compounded fp32 drift than Remax's dense 10-way output.
ACC_TOL = 0.01
ACC_MIN = 0.55

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
    return x_train, y_train, x_test, y_test


# ──────────────────────────────────────────────────────────────────────────────
#  Weight init
# ──────────────────────────────────────────────────────────────────────────────


def _he_conv(C_in, C_out, k):
    fan_in = C_in * k * k
    scale = math.sqrt(1.0 / fan_in)
    K = C_in * k * k
    mw = torch.randn(K, C_out) * scale
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
    l1 = _load_linear(TLinear(256, HRC_LEN, device=DEVICE), lin_p[1])

    return TSequential(
        [
            c0, TReLU(), bn0, TAvgPool2D(2),
            c1, TReLU(), bn1, TAvgPool2D(2),
            c2, TReLU(), bn2, TAvgPool2D(2),
            TFlatten(), l0, TReLU(), l1,
        ],
        device=DEVICE,
    )


def _flat_conv(mw, Sw, mb, Sb):
    return (
        mw.T.cpu().numpy().flatten().tolist(),
        Sw.T.cpu().numpy().flatten().tolist(),
        mb.squeeze().cpu().numpy().tolist(),
        Sb.squeeze().cpu().numpy().tolist(),
    )


def _flat_lin(mw, Sw, mb, Sb):
    return (
        mw.T.cpu().numpy().flatten().tolist(),
        Sw.T.cpu().numpy().flatten().tolist(),
        mb.squeeze().cpu().numpy().tolist(),
        Sb.squeeze().cpu().numpy().tolist(),
    )


def _flat_bn(mg, Sg, mb, Sb):
    return (
        mg.cpu().numpy().tolist(),
        Sg.cpu().numpy().tolist(),
        mb.cpu().numpy().tolist(),
        Sb.cpu().numpy().tolist(),
    )


def _build_pytagi(conv_p, bn_p, lin_p):
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
        PLinear(256, HRC_LEN),
    )
    net.preinit_layer()

    def _by_idx(prefix):
        return sorted(
            (k for k in net.state_dict() if k.startswith(prefix)),
            key=lambda k: int(k.split(".")[-1]),
        )

    conv_keys = _by_idx("Conv2d")
    bn_keys = _by_idx("BatchNorm2d")
    lin_keys = _by_idx("Linear")

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


def _train_triton(net, tri_hrc, x_train, y_train, perm):
    x_s = x_train[perm]
    y_s = y_train[perm]
    net.train()
    for i in range(0, len(x_s) - (len(x_s) % BATCH), BATCH):
        xb = x_s[i : i + BATCH].to(DEVICE)
        lb = y_s[i : i + BATCH].to(DEVICE)
        net.step_hrc(xb, lb, tri_hrc, SIGMA_V)


def _train_pytagi(net, updater, utils, x_train, y_train, perm, n_obs):
    x_np = x_train[perm].numpy()
    y_np = y_train[perm].numpy().astype(np.int32)
    for i in range(0, len(x_np) - (len(x_np) % BATCH), BATCH):
        xb = x_np[i : i + BATCH].reshape(-1).astype(np.float32)
        lb = y_np[i : i + BATCH]
        nb = BATCH
        obs_np, obs_idx_np, _ = utils.label_to_obs(lb, N_CLASSES)
        var_yb = np.full(nb * n_obs, SIGMA_V ** 2, dtype=np.float32)
        net(xb)
        updater.update_using_indices(
            output_states=net.output_z_buffer,
            mu_obs=obs_np.astype(np.float32),
            var_obs=var_yb,
            selected_idx=obs_idx_np.astype(np.int32),
            delta_states=net.input_delta_z_buffer,
        )
        net.backward()
        net.step()


def _accuracy_triton(net, tri_hrc, x_test, y_labels):
    net.eval()
    correct = 0
    x = x_test.to(DEVICE)
    n = len(x) - (len(x) % BATCH)
    for i in range(0, n, BATCH):
        xb = x[i : i + BATCH]
        ma, Sa = net.forward(xb)
        preds = get_predicted_labels(ma, Sa, tri_hrc)
        correct += (preds.cpu() == y_labels[i : i + BATCH]).sum().item()
    return correct / n


def _accuracy_pytagi(net, metric, x_test, y_labels):
    correct = 0
    x_np = x_test.numpy()
    n = len(x_np) - (len(x_np) % BATCH)
    for i in range(0, n, BATCH):
        xb = x_np[i : i + BATCH].reshape(-1).astype(np.float32)
        ma_flat, Sa_flat = net(xb)
        preds = metric.get_predicted_labels(np.array(ma_flat), np.array(Sa_flat))
        correct += (torch.tensor(preds, dtype=torch.long) == y_labels[i : i + BATCH]).sum().item()
    return correct / n


# ──────────────────────────────────────────────────────────────────────────────
#  The test
# ──────────────────────────────────────────────────────────────────────────────


def test_cifar10_cnn_bn_hrc_10epochs():
    """10-epoch CIFAR-10 BN-CNN + HRC parity: triton-tagi (CUDA) vs cuTAGI (CUDA)."""
    torch.manual_seed(0)
    pytagi.manual_seed(0)   # cuTAGI has its own RNG; without this its CUDA reductions drift run-to-run
    conv_p = [_he_conv(IN_C, 32, 5), _he_conv(32, 64, 5), _he_conv(64, 64, 5)]
    bn_p = [_bn_init(32), _bn_init(64), _bn_init(64)]
    lin_p = [_he_linear(1024, 256), _he_linear(256, HRC_LEN)]

    tri_hrc = class_to_obs(N_CLASSES)
    utils = Utils()
    metric = HRCSoftmaxMetric(num_classes=N_CLASSES)

    x_train, y_train, x_test, y_test = _load_cifar10()

    net_tri = _build_triton(conv_p, bn_p, lin_p)
    net_cut = _build_pytagi(conv_p, bn_p, lin_p)
    updater = OutputUpdater(net_cut.device)

    for _epoch in range(N_EPOCHS):
        perm = torch.randperm(len(x_train))
        _train_triton(net_tri, tri_hrc, x_train, y_train, perm)
        _train_pytagi(net_cut, updater, utils, x_train, y_train, perm, tri_hrc.n_obs)

    acc_tri = _accuracy_triton(net_tri, tri_hrc, x_test, y_test)
    acc_cut = _accuracy_pytagi(net_cut, metric, x_test, y_test)

    print(f"\n  triton-tagi (CUDA) BN-CNN + HRC:  {acc_tri * 100:.2f}%")
    print(f"  cuTAGI (CUDA)     BN-CNN + HRC:  {acc_cut * 100:.2f}%")
    print(f"  Δ accuracy:                        {abs(acc_tri - acc_cut) * 100:.3f}%  (tol {ACC_TOL * 100:.1f}%)")

    assert acc_tri >= ACC_MIN, f"triton-tagi: {acc_tri * 100:.2f}% < {ACC_MIN * 100:.0f}%"
    assert acc_cut >= ACC_MIN, f"cuTAGI: {acc_cut * 100:.2f}% < {ACC_MIN * 100:.0f}%"
    assert abs(acc_tri - acc_cut) < ACC_TOL, (
        f"BN-CNN + HRC accuracy gap {abs(acc_tri - acc_cut) * 100:.3f}% > {ACC_TOL * 100:.1f}% "
        f"(tri={acc_tri * 100:.2f}%  cut={acc_cut * 100:.2f}%)"
    )
