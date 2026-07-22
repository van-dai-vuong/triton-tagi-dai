"""Validation: triton-tagi HRC softmax vs cuTAGI (pytagi).

Tests three levels:
  1. Tree construction  — class_to_obs matches cuTAGI Utils.get_hierarchical_softmax
  2. Class probabilities — obs_to_class_probs matches cuTAGI Utils.obs_to_label_prob
  3. End-to-end MNIST    — HRC-trained MLP reaches same accuracy as cuTAGI HRC

Run with:
    pytest tests/validation/test_hrc_softmax.py -v -s
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torchvision import datasets

pytagi = pytest.importorskip("pytagi", reason="cuTAGI (pytagi) not installed")
from pytagi import HRCSoftmaxMetric, Utils
from pytagi.nn import Linear as PLinear
from pytagi.nn import MixtureReLU, OutputUpdater
from pytagi.nn import Sequential as PSequential

from triton_tagi.hrc_softmax import (
    class_to_obs,
    get_predicted_labels,
    labels_to_hrc,
    obs_to_class_probs,
)
from triton_tagi.layers.linear import Linear as TLinear
from triton_tagi.layers.relu import ReLU as TReLU
from triton_tagi.network import Sequential as TSequential

pytestmark = pytest.mark.cuda

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATOL = 1e-4

# ──────────────────────────────────────────────────────────────────────────────
#  Level 1: Tree structure
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_classes", [2, 4, 8, 10, 16, 100])
def test_class_to_obs_matches_cutagi(n_classes: int):
    """obs and idx must match cuTAGI's hierarchical_softmax_wrapper exactly."""
    utils = Utils()
    cutagi_hrc = utils.get_hierarchical_softmax(n_classes)

    tri_hrc = class_to_obs(n_classes)

    L = math.ceil(math.log2(n_classes))
    assert tri_hrc.n_obs == L
    assert tri_hrc.n_obs == cutagi_hrc.num_obs

    # cuTAGI stores obs and idx as flat lists (row-major n_classes × L)
    cutagi_obs = torch.tensor(cutagi_hrc.obs, dtype=torch.float32).reshape(n_classes, L)
    cutagi_idx = torch.tensor(cutagi_hrc.idx, dtype=torch.int32).reshape(n_classes, L)

    torch.testing.assert_close(tri_hrc.obs, cutagi_obs, atol=ATOL, rtol=0)
    torch.testing.assert_close(
        tri_hrc.idx.float(), cutagi_idx.float(), atol=0, rtol=0
    )
    assert tri_hrc.len == cutagi_hrc.len


# ──────────────────────────────────────────────────────────────────────────────
#  Level 2: Class probabilities
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_classes", [4, 10, 16])
def test_obs_to_class_probs_matches_cutagi(n_classes: int):
    """obs_to_class_probs must match cuTAGI's obs_to_label_prob within ATOL."""
    torch.manual_seed(0)
    utils = Utils()
    cutagi_hrc = utils.get_hierarchical_softmax(n_classes)
    tri_hrc = class_to_obs(n_classes)

    B = 8
    ma = torch.randn(B, tri_hrc.len)
    Sa = torch.rand(B, tri_hrc.len).abs() * 0.1

    # triton-tagi
    P_tri = obs_to_class_probs(ma, Sa, tri_hrc, alpha=3.0)  # (B, n_classes)

    # cuTAGI
    P_cut = []
    for i in range(B):
        ma_i = ma[i].numpy().tolist()
        Sa_i = Sa[i].numpy().tolist()
        p = utils.obs_to_label_prob(
            np.array(ma_i, dtype=np.float32),
            np.array(Sa_i, dtype=np.float32),
            cutagi_hrc,
            n_classes,
        )
        P_cut.append(p)
    P_cut = torch.tensor(np.array(P_cut), dtype=torch.float32)

    torch.testing.assert_close(P_tri, P_cut, atol=ATOL, rtol=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Level 3: End-to-end MNIST with HRC
# ──────────────────────────────────────────────────────────────────────────────

_N_CLASSES = 10
_L = math.ceil(math.log2(_N_CLASSES))
_HRC_LEN = 11   # known for 10 classes
_IN_F = 784
_H = 128
_SIGMA_V = 0.05
_BATCH = 512
_N_EPOCHS = 3
_ACC_TOL = 0.015   # 1.5 percentage-point tolerance
_ACC_MIN = 0.90    # both must reach ≥ 90 %
DATA_ROOT = "data"


def _load_mnist():
    train_ds = datasets.MNIST(DATA_ROOT, train=True, download=False)
    test_ds = datasets.MNIST(DATA_ROOT, train=False, download=False)
    x_train = train_ds.data.float().view(-1, _IN_F) / 255.0
    x_test = test_ds.data.float().view(-1, _IN_F) / 255.0
    mu, sigma = x_train.mean(), x_train.std()
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma
    return x_train, train_ds.targets, x_test, test_ds.targets


def _he_init(fan_in, fan_out):
    scale = math.sqrt(1.0 / fan_in)
    mw = torch.randn(fan_in, fan_out) * scale
    Sw = torch.full((fan_in, fan_out), scale**2)
    mb = torch.randn(1, fan_out) * scale
    Sb = torch.full((1, fan_out), scale**2)
    return mw, Sw, mb, Sb


def _build_triton(p0, p1):
    mw0, Sw0, mb0, Sb0 = p0
    mw1, Sw1, mb1, Sb1 = p1
    l0 = TLinear(_IN_F, _H, device=DEVICE)
    l0.mw, l0.Sw, l0.mb, l0.Sb = mw0.to(DEVICE), Sw0.to(DEVICE), mb0.to(DEVICE), Sb0.to(DEVICE)
    l1 = TLinear(_H, _HRC_LEN, device=DEVICE)
    l1.mw, l1.Sw, l1.mb, l1.Sb = mw1.to(DEVICE), Sw1.to(DEVICE), mb1.to(DEVICE), Sb1.to(DEVICE)
    return TSequential([l0, TReLU(), l1], device=DEVICE)


def _build_pytagi(p0, p1):
    mw0, Sw0, mb0, Sb0 = p0
    mw1, Sw1, mb1, Sb1 = p1

    def _flat(mw, Sw, mb, Sb):
        return (
            mw.T.numpy().flatten().tolist(),
            Sw.T.numpy().flatten().tolist(),
            mb.squeeze().numpy().tolist(),
            Sb.squeeze().numpy().tolist(),
        )

    net = PSequential(PLinear(_IN_F, _H), MixtureReLU(), PLinear(_H, _HRC_LEN))
    net.preinit_layer()
    keys = sorted(net.state_dict().keys())
    net.load_state_dict({keys[0]: _flat(mw0, Sw0, mb0, Sb0), keys[1]: _flat(mw1, Sw1, mb1, Sb1)})
    return net


def test_mnist_hrc_3epochs():
    """Both implementations reach ≥ 90 % and are within 1.5 % of each other."""
    torch.manual_seed(0)
    params = [_he_init(_IN_F, _H), _he_init(_H, _HRC_LEN)]

    tri_hrc = class_to_obs(_N_CLASSES)
    utils = Utils()
    cut_hrc = utils.get_hierarchical_softmax(_N_CLASSES)
    metric = HRCSoftmaxMetric(num_classes=_N_CLASSES)

    x_train, y_labels_train, x_test, y_labels_test = _load_mnist()

    net_tri = _build_triton(*params)
    net_cut = _build_pytagi(*params)
    updater = OutputUpdater(net_cut.device)

    for epoch in range(_N_EPOCHS):
        perm = torch.randperm(len(x_train))
        x_s = x_train[perm]
        y_s = y_labels_train[perm]

        # ── triton-tagi ──
        net_tri.train()
        for i in range(0, len(x_s), _BATCH):
            xb = x_s[i : i + _BATCH].to(DEVICE)
            lb = y_s[i : i + _BATCH].to(DEVICE)
            net_tri.step_hrc(xb, lb, tri_hrc, _SIGMA_V)

        # ── cuTAGI ──
        x_np = x_s.numpy()
        y_np = y_s.numpy().astype(np.int32)
        var_y = np.full((_BATCH * tri_hrc.n_obs,), _SIGMA_V**2, dtype=np.float32)
        for i in range(0, len(x_np), _BATCH):
            xb = x_np[i : i + _BATCH].flatten().astype(np.float32)
            lb = y_np[i : i + _BATCH]
            nb = len(lb)
            obs_np, obs_idx_np, _ = utils.label_to_obs(lb, _N_CLASSES)
            var_yb = np.full(nb * tri_hrc.n_obs, _SIGMA_V**2, dtype=np.float32)
            net_cut(xb)
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
    for i in range(0, len(x_test_gpu), 1024):
        xb = x_test_gpu[i : i + 1024]
        ma, Sa = net_tri.forward(xb)
        preds = get_predicted_labels(ma, Sa, tri_hrc)
        correct_tri += (preds.cpu() == y_labels_test[i : i + 1024]).sum().item()
    acc_tri = correct_tri / len(y_labels_test)

    correct_cut = 0
    x_np = x_test.numpy()
    for i in range(0, len(x_np), 1024):
        xb = x_np[i : i + 1024]
        nb = len(xb)
        ma_flat, Sa_flat = net_cut(xb.flatten().astype(np.float32))
        m_pred = metric.get_predicted_labels(
            np.array(ma_flat), np.array(Sa_flat)
        )
        m_pred_tensor = torch.tensor(m_pred, dtype=torch.long)
        correct_cut += (m_pred_tensor == y_labels_test[i : i + nb]).sum().item()
    acc_cut = correct_cut / len(y_labels_test)

    print(f"\n  triton-tagi HRC:  {acc_tri * 100:.2f}%")
    print(f"  cuTAGI HRC:       {acc_cut * 100:.2f}%")
    print(f"  Δ accuracy:       {abs(acc_tri - acc_cut) * 100:.3f}%  (tol {_ACC_TOL*100:.1f}%)")

    assert acc_tri >= _ACC_MIN, f"triton-tagi HRC: {acc_tri*100:.2f}% < {_ACC_MIN*100:.0f}%"
    assert acc_cut >= _ACC_MIN, f"cuTAGI HRC: {acc_cut*100:.2f}% < {_ACC_MIN*100:.0f}%"
    assert abs(acc_tri - acc_cut) < _ACC_TOL, (
        f"Accuracy gap {abs(acc_tri - acc_cut)*100:.3f}% > {_ACC_TOL*100:.1f}%"
        f"  tri={acc_tri*100:.2f}%  cut={acc_cut*100:.2f}%"
    )
