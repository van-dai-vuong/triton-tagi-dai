"""Validation: triton-tagi MaxPool2D vs cuTAGI (pytagi).

Two levels:
  1. Forward formula — given same weights and input, triton Conv2D→MaxPool2D
     produces the same (mu_a, var_a) as cuTAGI's pipeline.
  2. End-to-end CIFAR-10 CNN — same as test_cifar10_cnn but with MaxPool2D
     instead of AvgPool2D; verifies backward + parameter update through
     the max-pooling layer.

Architecture for Level 2:
    Conv2D(3, 8, 3, pad=1) → ReLU → MaxPool2D(4, 4) →
    Conv2D(8, 16, 3, pad=1) → ReLU → MaxPool2D(2, 2) →
    Flatten → Linear(256, 11)   [HRC output]

Run with:
    pytest tests/validation/test_maxpool2d.py -v -s
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
from pytagi.nn import Conv2d as PConv2d
from pytagi.nn import Linear as PLinear
from pytagi.nn import MaxPool2d as PMaxPool2d
from pytagi.nn import OutputUpdater
from pytagi.nn import ReLU as PReLU
from pytagi.nn import Sequential as PSequential

from triton_tagi.hrc_softmax import class_to_obs, get_predicted_labels, labels_to_hrc
from triton_tagi.layers.avgpool2d import AvgPool2D as TAvgPool2D
from triton_tagi.layers.conv2d import Conv2D as TConv2D
from triton_tagi.layers.flatten import Flatten as TFlatten
from triton_tagi.layers.linear import Linear as TLinear
from triton_tagi.layers.maxpool2d import MaxPool2D as TMaxPool2D
from triton_tagi.layers.relu import ReLU as TReLU
from triton_tagi.network import Sequential as TSequential

pytestmark = pytest.mark.cuda

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATOL = 1e-4
DATA_ROOT = "data"

# ──────────────────────────────────────────────────────────────────────────────
#  Level 1: Forward formula
# ──────────────────────────────────────────────────────────────────────────────


def _flat_conv_weights(mw, Sw, mb, Sb):
    """Convert triton Conv2D (K, C_out) layout to pytagi flat (C_out × K) format."""
    return (
        mw.T.numpy().flatten().tolist(),
        Sw.T.numpy().flatten().tolist(),
        mb.squeeze().numpy().tolist(),
        Sb.squeeze().numpy().tolist(),
    )


def test_maxpool2d_forward_matches_cutagi():
    """Triton Conv2D → MaxPool2D matches cuTAGI forward numerically."""
    torch.manual_seed(0)
    N, C_in, H, W, C_out, k = 4, 3, 8, 8, 8, 3
    pool_k, pool_s = 2, 2

    # ── Shared conv weights ──
    K = C_in * k * k
    scale = math.sqrt(1.0 / (K))
    mw = torch.randn(K, C_out) * scale
    Sw = torch.full((K, C_out), scale**2)
    mb = torch.zeros(1, C_out)
    Sb = torch.full((1, C_out), scale**2)

    # ── Input (Sa = 0 so both see same Sa_in) ──
    x_np = torch.randn(N, C_in, H, W).numpy().astype(np.float32)

    # ── triton forward ──
    tri_conv = TConv2D(C_in, C_out, k, padding=1, device=DEVICE)
    tri_conv.mw = mw.to(DEVICE)
    tri_conv.Sw = Sw.to(DEVICE)
    tri_conv.mb = mb.to(DEVICE)
    tri_conv.Sb = Sb.to(DEVICE)
    tri_pool = TMaxPool2D(pool_k, stride=pool_s)

    x_tri = torch.tensor(x_np, device=DEVICE)
    Sa_in = torch.zeros_like(x_tri)
    mz_conv, Sz_conv = tri_conv.forward(x_tri, Sa_in)
    mz_pool, Sz_pool = tri_pool.forward(mz_conv, Sz_conv)

    # ── pytagi forward ──
    net = PSequential(
        PConv2d(C_in, C_out, k, padding=1, in_width=W, in_height=H),
        PMaxPool2d(pool_k, pool_s),
    )
    net.preinit_layer()
    sd = net.state_dict()
    conv_key = [k2 for k2 in sd.keys() if "Conv" in k2][0]
    net.load_state_dict({conv_key: _flat_conv_weights(mw, Sw, mb, Sb)})

    x_flat = x_np.reshape(-1)
    ma_cut_flat, Sa_cut_flat = net(x_flat)
    H_out, W_out = H // pool_s, W // pool_s
    mz_cut = torch.tensor(np.array(ma_cut_flat)).reshape(N, C_out, H_out, W_out)
    Sz_cut = torch.tensor(np.array(Sa_cut_flat)).reshape(N, C_out, H_out, W_out)

    torch.testing.assert_close(mz_pool.cpu(), mz_cut, atol=ATOL, rtol=0)
    torch.testing.assert_close(Sz_pool.cpu(), Sz_cut, atol=ATOL, rtol=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Level 2: CIFAR-10 CNN with MaxPool2D
# ──────────────────────────────────────────────────────────────────────────────

N_CLASSES = 10
HRC_LEN = 11
IN_C, H_IMG, W_IMG = 3, 32, 32
SIGMA_V = 0.05
BATCH = 512
N_EPOCHS = 3
ACC_MIN = 0.30    # both must beat random (10 %) by a wide margin
ACC_TOL = 0.05    # 5 percentage points — MaxPool has more variance than AvgPool

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def _load_cifar10():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_ds = datasets.CIFAR10(DATA_ROOT, train=True, download=False, transform=tf)
    test_ds = datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=tf)
    x_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    y_train = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
    x_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    y_test = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
    return x_train, y_train, x_test, y_test


def _he_conv(C_in, C_out, k):
    fan_in = C_in * k * k
    scale = math.sqrt(1.0 / fan_in)
    K = C_in * k * k
    mw = torch.randn(K, C_out) * scale
    Sw = torch.full((K, C_out), scale**2)
    mb = torch.zeros(1, C_out)
    Sb = torch.full((1, C_out), scale**2)
    return mw, Sw, mb, Sb


def _he_linear(fan_in, fan_out):
    scale = math.sqrt(1.0 / fan_in)
    mw = torch.randn(fan_in, fan_out) * scale
    Sw = torch.full((fan_in, fan_out), scale**2)
    mb = torch.zeros(1, fan_out)
    Sb = torch.full((1, fan_out), scale**2)
    return mw, Sw, mb, Sb


def _build_triton(p_conv0, p_conv1, p_lin):
    mw0, Sw0, mb0, Sb0 = p_conv0
    mw1, Sw1, mb1, Sb1 = p_conv1
    mw2, Sw2, mb2, Sb2 = p_lin

    c0 = TConv2D(IN_C, 8, 3, padding=1, device=DEVICE)
    c0.mw, c0.Sw = mw0.to(DEVICE), Sw0.to(DEVICE)
    c0.mb, c0.Sb = mb0.to(DEVICE), Sb0.to(DEVICE)

    c1 = TConv2D(8, 16, 3, padding=1, device=DEVICE)
    c1.mw, c1.Sw = mw1.to(DEVICE), Sw1.to(DEVICE)
    c1.mb, c1.Sb = mb1.to(DEVICE), Sb1.to(DEVICE)

    l0 = TLinear(256, HRC_LEN, device=DEVICE)
    l0.mw, l0.Sw = mw2.to(DEVICE), Sw2.to(DEVICE)
    l0.mb, l0.Sb = mb2.to(DEVICE), Sb2.to(DEVICE)

    return TSequential(
        [c0, TReLU(), TMaxPool2D(4, stride=4),
         c1, TReLU(), TMaxPool2D(2, stride=2),
         TFlatten(), l0],
        device=DEVICE,
    )


def _build_pytagi(p_conv0, p_conv1, p_lin):
    mw0, Sw0, mb0, Sb0 = p_conv0
    mw1, Sw1, mb1, Sb1 = p_conv1
    mw2, Sw2, mb2, Sb2 = p_lin

    net = PSequential(
        PConv2d(IN_C, 8, 3, padding=1, in_width=W_IMG, in_height=H_IMG),
        PReLU(),
        PMaxPool2d(4, 4),
        PConv2d(8, 16, 3, padding=1),
        PReLU(),
        PMaxPool2d(2, 2),
        PLinear(256, HRC_LEN),
    )
    net.preinit_layer()
    sd = net.state_dict()
    keys = sorted(sd.keys())

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

    # Keys: sorted — Conv2d.0, Conv2d.3, Linear.6
    conv_keys = [k for k in keys if "Conv2d" in k]
    lin_keys = [k for k in keys if "Linear" in k]
    net.load_state_dict({
        conv_keys[0]: _flat_conv(mw0, Sw0, mb0, Sb0),
        conv_keys[1]: _flat_conv(mw1, Sw1, mb1, Sb1),
        lin_keys[0]: _flat_lin(mw2, Sw2, mb2, Sb2),
    })
    net.to_device("cuda")
    return net


def test_cifar10_maxpool2d_3epochs():
    """Both implementations reach ≥ 30 % and are within 5 % of each other."""
    torch.manual_seed(0)
    params = [
        _he_conv(IN_C, 8, 3),
        _he_conv(8, 16, 3),
        _he_linear(256, HRC_LEN),
    ]

    tri_hrc = class_to_obs(N_CLASSES)
    utils = Utils()
    metric = HRCSoftmaxMetric(num_classes=N_CLASSES)

    x_train, y_train, x_test, y_test = _load_cifar10()

    net_tri = _build_triton(*params)
    net_cut = _build_pytagi(*params)
    updater = OutputUpdater(net_cut.device)

    for epoch in range(N_EPOCHS):
        perm = torch.randperm(len(x_train))
        x_s = x_train[perm]
        y_s = y_train[perm]

        # ── triton-tagi ──
        net_tri.train()
        for i in range(0, len(x_s), BATCH):
            xb = x_s[i : i + BATCH].to(DEVICE)
            lb = y_s[i : i + BATCH].to(DEVICE)
            net_tri.step_hrc(xb, lb, tri_hrc, SIGMA_V)

        # ── cuTAGI ──
        x_np = x_s.numpy()
        y_np = y_s.numpy().astype(np.int32)
        for i in range(0, len(x_np), BATCH):
            xb_np = x_np[i : i + BATCH]
            lb_np = y_np[i : i + BATCH]
            nb = len(lb_np)
            xb_flat = xb_np.reshape(-1).astype(np.float32)
            obs_np, obs_idx_np, _ = utils.label_to_obs(lb_np, N_CLASSES)
            var_yb = np.full(nb * tri_hrc.n_obs, SIGMA_V**2, dtype=np.float32)
            net_cut(xb_flat)
            updater.update_using_indices(
                output_states=net_cut.output_z_buffer,
                mu_obs=obs_np.astype(np.float32),
                var_obs=var_yb,
                selected_idx=obs_idx_np.astype(np.int32),
                delta_states=net_cut.input_delta_z_buffer,
            )
            net_cut.backward()
            net_cut.step()

    # ── Accuracy ──
    net_tri.eval()
    correct_tri = 0
    x_test_gpu = x_test.to(DEVICE)
    for i in range(0, len(x_test_gpu), 512):
        xb = x_test_gpu[i : i + 512]
        ma, Sa = net_tri.forward(xb)
        preds = get_predicted_labels(ma, Sa, tri_hrc)
        correct_tri += (preds.cpu() == y_test[i : i + 512]).sum().item()
    acc_tri = correct_tri / len(y_test)

    correct_cut = 0
    x_np = x_test.numpy()
    for i in range(0, len(x_np), 512):
        xb_np = x_np[i : i + 512]
        nb = len(xb_np)
        ma_flat, Sa_flat = net_cut(xb_np.reshape(-1).astype(np.float32))
        preds = metric.get_predicted_labels(np.array(ma_flat), np.array(Sa_flat))
        correct_cut += (torch.tensor(preds, dtype=torch.long) == y_test[i : i + nb]).sum().item()
    acc_cut = correct_cut / len(y_test)

    print(f"\n  triton-tagi MaxPool CNN:  {acc_tri * 100:.2f}%")
    print(f"  cuTAGI MaxPool CNN:       {acc_cut * 100:.2f}%")
    print(f"  Δ accuracy:                {abs(acc_tri - acc_cut) * 100:.3f}%  (tol {ACC_TOL*100:.1f}%)")

    assert acc_tri >= ACC_MIN, f"triton-tagi: {acc_tri*100:.2f}% < {ACC_MIN*100:.0f}%"
    assert acc_cut >= ACC_MIN, f"cuTAGI: {acc_cut*100:.2f}% < {ACC_MIN*100:.0f}%"
    assert abs(acc_tri - acc_cut) < ACC_TOL, (
        f"gap {abs(acc_tri - acc_cut)*100:.3f}% > {ACC_TOL*100:.1f}%  "
        f"tri={acc_tri*100:.2f}%  cut={acc_cut*100:.2f}%"
    )
