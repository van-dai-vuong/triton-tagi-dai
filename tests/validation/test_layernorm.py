"""Validation: triton-tagi LayerNorm vs cuTAGI (pytagi).

Two levels:
  1. Forward formula — given same inputs and weights, triton LayerNorm produces
     the same (mz, Sz) as the cuTAGI CPU reference formula in fp64.
  2. End-to-end MNIST — MLP with LayerNorm trained for 3 epochs; both
     implementations must reach ≥ 90 % and be within 1.5 % of each other.

Architecture for Level 2:
    Linear(784, 128) → LayerNorm(128) → ReLU → Linear(128, 11)  [HRC output]

Run with:
    pytest tests/validation/test_layernorm.py -v -s
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torchvision import datasets

pytagi = pytest.importorskip("pytagi", reason="cuTAGI (pytagi) not installed")
from pytagi import HRCSoftmaxMetric, Utils
from pytagi.nn import LayerNorm as PLayerNorm
from pytagi.nn import Linear as PLinear
from pytagi.nn import MixtureReLU, OutputUpdater
from pytagi.nn import Sequential as PSequential

from triton_tagi.hrc_softmax import class_to_obs, get_predicted_labels, labels_to_hrc
from triton_tagi.layers.layernorm import LayerNorm as TLayerNorm
from triton_tagi.layers.linear import Linear as TLinear
from triton_tagi.layers.relu import ReLU as TReLU
from triton_tagi.network import Sequential as TSequential

pytestmark = pytest.mark.cuda

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATOL = 1e-4
DATA_ROOT = "data"

# ──────────────────────────────────────────────────────────────────────────────
#  Level 1: Forward formula
# ──────────────────────────────────────────────────────────────────────────────


def _ln_fwd_ref(ma, Sa, mw, Sw, mb, Sb, eps=1e-5):
    """fp64 reference for cuTAGI's LayerNorm forward formula."""
    ma64, Sa64 = ma.double(), Sa.double()
    mw64, Sw64 = mw.double(), Sw.double()
    mb64, Sb64 = mb.double(), Sb.double()
    ni = ma64.shape[1]

    mu_ra = ma64.mean(dim=1, keepdim=True)
    var_s = Sa64.sum(dim=1, keepdim=True)
    mu_diff = ma64 - mu_ra
    var_ra = (mu_diff**2).sum(dim=1, keepdim=True) + var_s
    var_ra = var_ra / (ni - 1)

    inv_std = 1.0 / (var_ra + eps).sqrt()
    mz = inv_std * mu_diff * mw64 + mb64
    Sz = inv_std**2 * (Sa64 * (mw64**2 + Sw64) + Sw64 * mu_diff**2) + Sb64
    return mz.float(), Sz.float()


@pytest.mark.parametrize("ni", [8, 32, 128])
def test_layernorm_forward_mean(ni: int):
    """LayerNorm forward mean matches fp64 reference."""
    torch.manual_seed(0)
    B = 16
    ma = torch.randn(B, ni)
    Sa = torch.rand(B, ni).abs() * 0.1 + 1e-4
    mw = torch.ones(ni) + torch.randn(ni) * 0.1
    Sw = torch.rand(ni).abs() * 0.01 + 1e-5
    mb = torch.randn(ni) * 0.1
    Sb = torch.rand(ni).abs() * 0.01 + 1e-5

    layer = TLayerNorm(ni, device=DEVICE)
    layer.mw = mw.to(DEVICE)
    layer.Sw = Sw.to(DEVICE)
    layer.mb = mb.to(DEVICE)
    layer.Sb = Sb.to(DEVICE)
    mz_tri, _ = layer.forward(ma.to(DEVICE), Sa.to(DEVICE))

    mz_ref, _ = _ln_fwd_ref(ma, Sa, mw, Sw, mb, Sb)
    torch.testing.assert_close(mz_tri.cpu(), mz_ref, atol=ATOL, rtol=0)


@pytest.mark.parametrize("ni", [8, 32, 128])
def test_layernorm_forward_variance(ni: int):
    """LayerNorm forward variance matches fp64 reference."""
    torch.manual_seed(1)
    B = 16
    ma = torch.randn(B, ni)
    Sa = torch.rand(B, ni).abs() * 0.1 + 1e-4
    mw = torch.ones(ni) + torch.randn(ni) * 0.1
    Sw = torch.rand(ni).abs() * 0.01 + 1e-5
    mb = torch.randn(ni) * 0.1
    Sb = torch.rand(ni).abs() * 0.01 + 1e-5

    layer = TLayerNorm(ni, device=DEVICE)
    layer.mw = mw.to(DEVICE)
    layer.Sw = Sw.to(DEVICE)
    layer.mb = mb.to(DEVICE)
    layer.Sb = Sb.to(DEVICE)
    _, Sz_tri = layer.forward(ma.to(DEVICE), Sa.to(DEVICE))

    _, Sz_ref = _ln_fwd_ref(ma, Sa, mw, Sw, mb, Sb)
    torch.testing.assert_close(Sz_tri.cpu(), Sz_ref, atol=ATOL, rtol=0)


def test_layernorm_forward_matches_cutagi():
    """Triton LayerNorm forward numerics match cuTAGI's pytagi (via identity linear)."""
    torch.manual_seed(42)
    ni = 16
    B = 8

    # Shared gamma/beta
    mw = torch.ones(ni)
    Sw = torch.full((ni,), 1e-5)
    mb = torch.zeros(ni)
    Sb = torch.full((ni,), 1e-5)

    # ── triton: apply LayerNorm directly ──
    layer = TLayerNorm(ni, device=DEVICE)
    layer.mw = mw.to(DEVICE)
    layer.Sw = Sw.to(DEVICE)
    layer.mb = mb.to(DEVICE)
    layer.Sb = Sb.to(DEVICE)

    ma = torch.randn(B, ni)
    Sa = torch.zeros(B, ni)  # zero Sa so cuTAGI identity-linear doesn't add noise
    mz_tri, Sz_tri = layer.forward(ma.to(DEVICE), Sa.to(DEVICE))

    # ── pytagi: identity Linear(ni, ni) + LayerNorm([ni]) ──
    # Identity linear: mw=I, Sw≈0, mb=0, Sb≈0 to pass input unchanged with ~0 Sa
    net = PSequential(PLinear(ni, ni), PLayerNorm([ni]))
    net.preinit_layer()
    sd = net.state_dict()
    keys = sorted(sd.keys())   # ['LayerNorm.1', 'Linear.0']
    ln_key = [k for k in keys if "LayerNorm" in k][0]
    lin_key = [k for k in keys if "Linear" in k][0]

    identity_mw = torch.eye(ni).T.numpy().flatten().tolist()  # pytagi: (out, in) -> transpose of I = I
    net.load_state_dict({
        lin_key: (identity_mw, [1e-10] * (ni * ni), [0.0] * ni, [1e-10] * ni),
        ln_key:  (mw.tolist(), Sw.tolist(), mb.tolist(), Sb.tolist()),
    })

    x_flat = ma.numpy().reshape(-1).astype(np.float32)
    ma_cut_flat, Sa_cut_flat = net(x_flat)
    mz_cut = torch.tensor(np.array(ma_cut_flat)).reshape(B, ni)
    Sz_cut = torch.tensor(np.array(Sa_cut_flat)).reshape(B, ni)

    # The identity linear introduces fp32 matmul rounding (~1e-7 per element),
    # which gets amplified slightly by the LayerNorm inv_std. Use 5×ATOL here;
    # the exact formula is validated to fp64 precision in test_layernorm_forward_*.
    torch.testing.assert_close(mz_tri.cpu(), mz_cut, atol=5 * ATOL, rtol=0)
    torch.testing.assert_close(Sz_tri.cpu(), Sz_cut, atol=5 * ATOL, rtol=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Level 2: End-to-end MNIST with LayerNorm
# ──────────────────────────────────────────────────────────────────────────────

_N_CLASSES = 10
_HRC_LEN = 11
_IN_F = 784
_H = 128
_SIGMA_V = 0.05
_BATCH = 512
_N_EPOCHS = 3
_ACC_TOL = 0.015   # 1.5 percentage-point tolerance
_ACC_MIN = 0.90    # both must reach ≥ 90 %


def _load_mnist():
    train_ds = datasets.MNIST(DATA_ROOT, train=True, download=False)
    test_ds = datasets.MNIST(DATA_ROOT, train=False, download=False)
    x_train = train_ds.data.float().view(-1, _IN_F) / 255.0
    x_test = test_ds.data.float().view(-1, _IN_F) / 255.0
    mu, sigma = x_train.mean(), x_train.std()
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma
    return x_train, train_ds.targets, x_test, test_ds.targets


def _he_linear(fan_in, fan_out):
    scale = math.sqrt(1.0 / fan_in)
    mw = torch.randn(fan_in, fan_out) * scale
    Sw = torch.full((fan_in, fan_out), scale**2)
    mb = torch.randn(1, fan_out) * scale
    Sb = torch.full((1, fan_out), scale**2)
    return mw, Sw, mb, Sb


def _he_norm(ni):
    """Init LayerNorm params: gamma=1, beta=0, Sw=Sb=scale."""
    scale = 2.0 / (ni + ni)
    mw = torch.ones(ni)
    Sw = torch.full((ni,), scale)
    mb = torch.zeros(ni)
    Sb = torch.full((ni,), scale)
    return mw, Sw, mb, Sb


def _build_triton(p_lin0, p_ln, p_lin1):
    mw0, Sw0, mb0, Sb0 = p_lin0
    mw_ln, Sw_ln, mb_ln, Sb_ln = p_ln
    mw1, Sw1, mb1, Sb1 = p_lin1

    l0 = TLinear(_IN_F, _H, device=DEVICE)
    l0.mw, l0.Sw, l0.mb, l0.Sb = mw0.to(DEVICE), Sw0.to(DEVICE), mb0.to(DEVICE), Sb0.to(DEVICE)

    ln = TLayerNorm(_H, device=DEVICE)
    ln.mw, ln.Sw, ln.mb, ln.Sb = mw_ln.to(DEVICE), Sw_ln.to(DEVICE), mb_ln.to(DEVICE), Sb_ln.to(DEVICE)

    l1 = TLinear(_H, _HRC_LEN, device=DEVICE)
    l1.mw, l1.Sw, l1.mb, l1.Sb = mw1.to(DEVICE), Sw1.to(DEVICE), mb1.to(DEVICE), Sb1.to(DEVICE)

    return TSequential([l0, ln, TReLU(), l1], device=DEVICE)


def _build_pytagi(p_lin0, p_ln, p_lin1):
    mw0, Sw0, mb0, Sb0 = p_lin0
    mw_ln, Sw_ln, mb_ln, Sb_ln = p_ln
    mw1, Sw1, mb1, Sb1 = p_lin1

    net = PSequential(
        PLinear(_IN_F, _H),
        PLayerNorm([_H]),
        MixtureReLU(),
        PLinear(_H, _HRC_LEN),
    )
    net.preinit_layer()
    sd = net.state_dict()
    keys = sorted(sd.keys())   # keys are: LayerNorm.1, Linear.0, Linear.3

    lin0_key = [k for k in keys if "Linear" in k][0]   # first Linear
    ln_key   = [k for k in keys if "LayerNorm" in k][0]
    lin1_key = [k for k in keys if "Linear" in k][1]   # second Linear

    def _flat_lin(mw, Sw, mb, Sb):
        return (
            mw.T.numpy().flatten().tolist(),
            Sw.T.numpy().flatten().tolist(),
            mb.squeeze().numpy().tolist(),
            Sb.squeeze().numpy().tolist(),
        )

    def _flat_ln(mw, Sw, mb, Sb):
        # LayerNorm in pytagi: weights are (ni,) — no transposition needed
        return (
            mw.numpy().tolist(),
            Sw.numpy().tolist(),
            mb.numpy().tolist(),
            Sb.numpy().tolist(),
        )

    net.load_state_dict({
        lin0_key: _flat_lin(mw0, Sw0, mb0, Sb0),
        ln_key:   _flat_ln(mw_ln, Sw_ln, mb_ln, Sb_ln),
        lin1_key: _flat_lin(mw1, Sw1, mb1, Sb1),
    })
    return net


def test_mnist_layernorm_3epochs():
    """Both implementations reach ≥ 90 % and are within 1.5 % of each other."""
    torch.manual_seed(0)
    params_lin0 = _he_linear(_IN_F, _H)
    params_ln   = _he_norm(_H)
    params_lin1 = _he_linear(_H, _HRC_LEN)

    tri_hrc = class_to_obs(_N_CLASSES)
    utils = Utils()
    metric = HRCSoftmaxMetric(num_classes=_N_CLASSES)

    x_train, y_train, x_test, y_test = _load_mnist()

    net_tri = _build_triton(params_lin0, params_ln, params_lin1)
    net_cut = _build_pytagi(params_lin0, params_ln, params_lin1)
    updater = OutputUpdater(net_cut.device)

    for epoch in range(_N_EPOCHS):
        perm = torch.randperm(len(x_train))
        x_s = x_train[perm]
        y_s = y_train[perm]

        # ── triton-tagi ──
        net_tri.train()
        for i in range(0, len(x_s), _BATCH):
            xb = x_s[i : i + _BATCH].to(DEVICE)
            lb = y_s[i : i + _BATCH].to(DEVICE)
            net_tri.step_hrc(xb, lb, tri_hrc, _SIGMA_V)

        # ── cuTAGI ──
        x_np = x_s.numpy()
        y_np = y_s.numpy().astype(np.int32)
        for i in range(0, len(x_np), _BATCH):
            xb_np = x_np[i : i + _BATCH]
            lb_np = y_np[i : i + _BATCH]
            nb = len(lb_np)
            xb_flat = xb_np.reshape(-1).astype(np.float32)
            obs_np, obs_idx_np, _ = utils.label_to_obs(lb_np, _N_CLASSES)
            var_yb = np.full(nb * tri_hrc.n_obs, _SIGMA_V**2, dtype=np.float32)
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
    for i in range(0, len(x_test_gpu), 1024):
        xb = x_test_gpu[i : i + 1024]
        ma, Sa = net_tri.forward(xb)
        preds = get_predicted_labels(ma, Sa, tri_hrc)
        correct_tri += (preds.cpu() == y_test[i : i + 1024]).sum().item()
    acc_tri = correct_tri / len(y_test)

    correct_cut = 0
    x_np = x_test.numpy()
    for i in range(0, len(x_np), 1024):
        xb_np = x_np[i : i + 1024]
        nb = len(xb_np)
        ma_flat, Sa_flat = net_cut(xb_np.flatten().astype(np.float32))
        preds = metric.get_predicted_labels(np.array(ma_flat), np.array(Sa_flat))
        preds_t = torch.tensor(preds, dtype=torch.long)
        correct_cut += (preds_t == y_test[i : i + nb]).sum().item()
    acc_cut = correct_cut / len(y_test)

    print(f"\n  triton-tagi LN:  {acc_tri * 100:.2f}%")
    print(f"  cuTAGI LN:       {acc_cut * 100:.2f}%")
    print(f"  Δ accuracy:       {abs(acc_tri - acc_cut) * 100:.3f}%  (tol {_ACC_TOL*100:.1f}%)")

    assert acc_tri >= _ACC_MIN, f"triton-tagi LN: {acc_tri*100:.2f}% < {_ACC_MIN*100:.0f}%"
    assert acc_cut >= _ACC_MIN, f"cuTAGI LN: {acc_cut*100:.2f}% < {_ACC_MIN*100:.0f}%"
    assert abs(acc_tri - acc_cut) < _ACC_TOL, (
        f"gap {abs(acc_tri - acc_cut)*100:.3f}% > {_ACC_TOL*100:.1f}%  "
        f"tri={acc_tri*100:.2f}%  cut={acc_cut*100:.2f}%"
    )
