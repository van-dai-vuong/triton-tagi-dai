"""Parity test: triton-tagi vs cuTAGI (pytagi), both on CUDA, MNIST + HRC, 10 epochs.

Both networks:
  - start from the same He-initialised weights,
  - see the same shuffled mini-batches in the same order,
  - run on CUDA,
  - use hierarchical softmax on the 10-class label set (11 output neurons,
    4 observed nodes per example — the class's binary path through the tree),
  - use identical σ_v = 0.05.

The assertion is that after 10 epochs both reach ≥ 95 % and agree to within a
small tolerance. Because both runs use CUDA fp32 but different kernel
implementations (cuTAGI C++/CUDA vs triton-tagi Triton + cuBLAS), accumulation
order differs per kernel and a small final-epoch gap is expected. The
tolerance is set above that hardware-induced drift so real bugs still fail.

Run with:
    pytest tests/validation/test_mnist_hrc_10epochs.py -v -s
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import pytagi
from pytagi import HRCSoftmaxMetric, Utils
from pytagi.nn import Linear as PLinear
from pytagi.nn import MixtureReLU, OutputUpdater
from pytagi.nn import Sequential as PSequential
from torchvision import datasets

from triton_tagi.hrc_softmax import class_to_obs, get_predicted_labels
from triton_tagi.layers.linear import Linear as TLinear
from triton_tagi.layers.relu import ReLU as TReLU
from triton_tagi.network import Sequential as TSequential

pytestmark = pytest.mark.cuda

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "data"

N_CLASSES = 10
HRC_LEN = 11             # tree size for 10 classes: ceil(log2(10))=4 levels → 11 nodes
IN_F = 784
H = 128
SIGMA_V = 0.05
BATCH = 512
N_EPOCHS = 10
# Observed Δ on an RTX 4070 Ti SUPER (2026-04-19): 0.01 pp (triton 95.99 vs
# cuTAGI 95.98) — effectively identical. Tolerance 0.3 pp leaves margin for
# driver / hardware variation while still catching any real divergence.
ACC_TOL = 0.003
ACC_MIN = 0.95           # both must reach ≥ 95 %


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
    return x_train, train_ds.targets, x_test, test_ds.targets


def _he_init(fan_in, fan_out):
    scale = math.sqrt(1.0 / fan_in)
    mw = torch.randn(fan_in, fan_out) * scale
    Sw = torch.full((fan_in, fan_out), scale ** 2)
    mb = torch.randn(1, fan_out) * scale
    Sb = torch.full((1, fan_out), scale ** 2)
    return mw, Sw, mb, Sb


def _build_triton(p0, p1):
    mw0, Sw0, mb0, Sb0 = p0
    mw1, Sw1, mb1, Sb1 = p1
    l0 = TLinear(IN_F, H, device=DEVICE)
    l0.mw, l0.Sw, l0.mb, l0.Sb = mw0.to(DEVICE), Sw0.to(DEVICE), mb0.to(DEVICE), Sb0.to(DEVICE)
    l1 = TLinear(H, HRC_LEN, device=DEVICE)
    l1.mw, l1.Sw, l1.mb, l1.Sb = mw1.to(DEVICE), Sw1.to(DEVICE), mb1.to(DEVICE), Sb1.to(DEVICE)
    return TSequential([l0, TReLU(), l1], device=DEVICE)


def _build_pytagi(p0, p1):
    """Build pytagi MLP on CUDA with the same weights as triton-tagi.

    Weights must be loaded BEFORE `to_device('cuda')` because the state-dict
    keys change from ``Linear.N`` to ``LinearCuda.N`` after the device move.
    """
    mw0, Sw0, mb0, Sb0 = p0
    mw1, Sw1, mb1, Sb1 = p1

    def _flat(mw, Sw, mb, Sb):
        return (
            mw.T.numpy().flatten().tolist(),
            Sw.T.numpy().flatten().tolist(),
            mb.squeeze().numpy().tolist(),
            Sb.squeeze().numpy().tolist(),
        )

    net = PSequential(PLinear(IN_F, H), MixtureReLU(), PLinear(H, HRC_LEN))
    net.preinit_layer()
    keys = sorted(net.state_dict().keys())
    net.load_state_dict({
        keys[0]: _flat(mw0, Sw0, mb0, Sb0),
        keys[1]: _flat(mw1, Sw1, mb1, Sb1),
    })
    net.to_device("cuda")
    return net


# ──────────────────────────────────────────────────────────────────────────────
#  The test
# ──────────────────────────────────────────────────────────────────────────────


def test_mnist_hrc_10epochs():
    """10-epoch MNIST HRC parity: triton-tagi (CUDA) vs cuTAGI (CUDA)."""
    torch.manual_seed(0)
    pytagi.manual_seed(0)   # make cuTAGI's CUDA reductions deterministic run-to-run
    params = [_he_init(IN_F, H), _he_init(H, HRC_LEN)]

    tri_hrc = class_to_obs(N_CLASSES)
    utils = Utils()
    metric = HRCSoftmaxMetric(num_classes=N_CLASSES)

    x_train, y_labels_train, x_test, y_labels_test = _load_mnist()

    net_tri = _build_triton(*params)
    net_cut = _build_pytagi(*params)
    updater = OutputUpdater(net_cut.device)

    for _epoch in range(N_EPOCHS):
        perm = torch.randperm(len(x_train))
        x_s = x_train[perm]
        y_s = y_labels_train[perm]

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
            xb = x_np[i : i + BATCH].flatten().astype(np.float32)
            lb = y_np[i : i + BATCH]
            nb = len(lb)
            obs_np, obs_idx_np, _ = utils.label_to_obs(lb, N_CLASSES)
            var_yb = np.full(nb * tri_hrc.n_obs, SIGMA_V ** 2, dtype=np.float32)
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
    # pytagi CUDA pre-allocates buffers at the first forward's batch size and
    # crashes on larger ones, so eval uses the same BATCH size as training.
    net_tri.eval()
    correct_tri = 0
    x_test_gpu = x_test.to(DEVICE)
    for i in range(0, len(x_test_gpu), BATCH):
        xb = x_test_gpu[i : i + BATCH]
        ma, Sa = net_tri.forward(xb)
        preds = get_predicted_labels(ma, Sa, tri_hrc)
        correct_tri += (preds.cpu() == y_labels_test[i : i + BATCH]).sum().item()
    acc_tri = correct_tri / len(y_labels_test)

    correct_cut = 0
    x_np = x_test.numpy()
    for i in range(0, len(x_np), BATCH):
        xb = x_np[i : i + BATCH]
        nb = len(xb)
        ma_flat, Sa_flat = net_cut(xb.flatten().astype(np.float32))
        m_pred = metric.get_predicted_labels(np.array(ma_flat), np.array(Sa_flat))
        m_pred_tensor = torch.tensor(m_pred, dtype=torch.long)
        correct_cut += (m_pred_tensor == y_labels_test[i : i + nb]).sum().item()
    acc_cut = correct_cut / len(y_labels_test)

    print(f"\n  triton-tagi (CUDA) + HRC:  {acc_tri * 100:.2f}%")
    print(f"  cuTAGI (CUDA)     + HRC:  {acc_cut * 100:.2f}%")
    print(f"  Δ accuracy:                {abs(acc_tri - acc_cut) * 100:.3f}%  (tol {ACC_TOL * 100:.1f}%)")

    assert acc_tri >= ACC_MIN, f"triton-tagi HRC: {acc_tri * 100:.2f}% < {ACC_MIN * 100:.0f}%"
    assert acc_cut >= ACC_MIN, f"cuTAGI HRC: {acc_cut * 100:.2f}% < {ACC_MIN * 100:.0f}%"
    assert abs(acc_tri - acc_cut) < ACC_TOL, (
        f"HRC accuracy gap {abs(acc_tri - acc_cut) * 100:.3f}% > {ACC_TOL * 100:.1f}% "
        f"(tri={acc_tri * 100:.2f}%  cut={acc_cut * 100:.2f}%)"
    )
