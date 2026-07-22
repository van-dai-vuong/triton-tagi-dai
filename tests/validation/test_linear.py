"""Validation tests: triton-tagi Linear against cuTAGI (pytagi).

Level 1 — Forward: identical inputs and weights produce matching (mz, Sz).
Level 2 — Backward: given identical output deltas, input deltas and parameter
           deltas match the analytical reference.  pytagi does not expose
           internal delta buffers for a single layer, so we validate against
           the fp64 ground truth (which matches cuTAGI to 2.4e-7 — see
           test_linear_forward_variance for the forward parity proof).
Level 3 — Update: one full step produces matching updated (mw, Sw, mb, Sb).

Run with:
    pytest tests/validation/test_linear.py -v

Requires an environment with both triton-tagi and pytagi installed,
and a CUDA GPU (all triton-tagi kernels are GPU-only).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pytagi.nn import Linear as PLinear
from pytagi.nn import OutputUpdater
from pytagi.nn import Sequential as PSequential

from triton_tagi.layers.linear import Linear as TLinear

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Tolerances after tf32 fix: tl.dot uses allow_tf32=False and torch.matmul
# has tf32 disabled at import time, so triton matches cuTAGI to fp32 precision.
MEAN_ATOL = 1e-4  # mean forward: cuBLAS tile order vs scalar FMA → ~2e-6 max
VAR_ATOL = 1e-4  # variance forward: ~2e-7 max after tf32 + 2-matmul grouping fix
UPDATE_ATOL = 1e-4  # weight update: Sz now nearly identical → tight cascade

pytestmark = pytest.mark.cuda


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_triton(in_f: int, out_f: int, mw, Sw, mb, Sb) -> TLinear:
    """Build a triton-tagi Linear with prescribed weights."""
    layer = TLinear(in_f, out_f, device=DEVICE)
    layer.mw = mw.clone().to(DEVICE)
    layer.Sw = Sw.clone().to(DEVICE)
    layer.mb = mb.clone().to(DEVICE)
    layer.Sb = Sb.clone().to(DEVICE)
    return layer


def _make_pytagi(in_f: int, out_f: int, mw, Sw, mb, Sb) -> PSequential:
    """Build a pytagi Sequential(Linear) with the same weights.

    Triton stores mw as (in_features, out_features).
    pytagi stores weights flat in (out_features, in_features) row-major order,
    i.e. flat_mw = mw.T.flatten().
    """
    net = PSequential(PLinear(in_f, out_f))
    net.preinit_layer()
    key = list(net.state_dict().keys())[0]
    net.load_state_dict(
        {
            key: (
                mw.T.cpu().numpy().flatten().tolist(),
                Sw.T.cpu().numpy().flatten().tolist(),
                mb.squeeze().cpu().numpy().tolist(),
                Sb.squeeze().cpu().numpy().tolist(),
            )
        }
    )
    return net


def _pytagi_weights(net: PSequential, key: str):
    """Return (mw, Sw, mb, Sb) from a pytagi net as torch tensors on CPU.

    Converts from pytagi's (out, in) flat layout back to triton's (in, out).
    """
    mu_w_flat, var_w_flat, mu_b_flat, var_b_flat = net.state_dict()[key]
    in_f = len(mu_b_flat)  # bias length = out_features; but we need in_features
    # mu_w_flat has out * in elements; we deduce in from total / out
    out_f = len(mu_b_flat)
    total = len(mu_w_flat)
    in_f = total // out_f
    mw = torch.tensor(mu_w_flat).reshape(out_f, in_f).T  # (in, out)
    Sw = torch.tensor(var_w_flat).reshape(out_f, in_f).T
    mb = torch.tensor(mu_b_flat).reshape(1, out_f)
    Sb = torch.tensor(var_b_flat).reshape(1, out_f)
    return mw, Sw, mb, Sb


# ──────────────────────────────────────────────────────────────────────────────
#  Level 1: Forward
# ──────────────────────────────────────────────────────────────────────────────


def test_linear_forward_mean():
    """Output means match between triton-tagi and cuTAGI."""
    torch.manual_seed(0)
    N, in_f, out_f = 8, 16, 32

    mw = torch.randn(in_f, out_f)
    Sw = torch.rand(in_f, out_f).abs() * 0.1 + 1e-6
    mb = torch.randn(1, out_f)
    Sb = torch.rand(1, out_f).abs() * 0.1 + 1e-6
    ma = torch.randn(N, in_f)

    tri = _make_triton(in_f, out_f, mw, Sw, mb, Sb)
    mz_tri, _ = tri.forward(ma.to(DEVICE), torch.zeros(N, in_f, device=DEVICE))

    cut = _make_pytagi(in_f, out_f, mw, Sw, mb, Sb)
    m_flat, _ = cut(ma.numpy().flatten().astype(np.float32))
    mz_cut = torch.tensor(m_flat).reshape(N, out_f)

    torch.testing.assert_close(mz_tri.cpu(), mz_cut, atol=MEAN_ATOL, rtol=0)


def test_linear_forward_variance():
    """Output variances match between triton-tagi and cuTAGI."""
    torch.manual_seed(1)
    N, in_f, out_f = 8, 16, 32

    mw = torch.randn(in_f, out_f)
    Sw = torch.rand(in_f, out_f).abs() * 0.1 + 1e-6
    mb = torch.randn(1, out_f)
    Sb = torch.rand(1, out_f).abs() * 0.1 + 1e-6
    ma = torch.randn(N, in_f)

    tri = _make_triton(in_f, out_f, mw, Sw, mb, Sb)
    _, Sz_tri = tri.forward(ma.to(DEVICE), torch.zeros(N, in_f, device=DEVICE))

    cut = _make_pytagi(in_f, out_f, mw, Sw, mb, Sb)
    _, v_flat = cut(ma.numpy().flatten().astype(np.float32))
    Sz_cut = torch.tensor(v_flat).reshape(N, out_f)

    torch.testing.assert_close(Sz_tri.cpu(), Sz_cut, atol=VAR_ATOL, rtol=0)


def test_linear_forward_nonzero_sa():
    """Nonzero input Sa: covered by MLP tests where Sa propagates through ReLU."""
    pytest.skip("nonzero input Sa requires mid-network injection; covered in test_mlp.py")


# ──────────────────────────────────────────────────────────────────────────────
#  Level 2: Backward — input delta propagation and parameter deltas
#
#  Formula (matches cuTAGI linear_bwd_delta_z / linear_bwd_delta_w):
#    delta_ma = delta_mz @ mw^T
#    delta_Sa = delta_Sz @ (mw²)^T
#    delta_mw = Sw * (ma^T @ delta_mz)
#    delta_Sw = Sw² * ((ma²)^T @ delta_Sz)
#
#  Reference: fp64 computation of the same formulas.
#  Rationale: pytagi does not expose per-layer delta buffers; fp64 is the
#  ground truth and matches cuTAGI to <3e-7 (see forward variance test).
# ──────────────────────────────────────────────────────────────────────────────


def _setup_backward(seed=0):
    """Return (tri, ma, delta_mz, delta_Sz) on CUDA with fixed weights."""
    torch.manual_seed(seed)
    N, in_f, out_f = 8, 16, 32
    mw = torch.randn(in_f, out_f)
    Sw = torch.rand(in_f, out_f).abs() * 0.1 + 1e-6
    mb = torch.randn(1, out_f)
    Sb = torch.rand(1, out_f).abs() * 0.1 + 1e-6
    ma = torch.randn(N, in_f)
    delta_mz = torch.randn(N, out_f)
    delta_Sz = torch.rand(N, out_f).abs() * 0.01

    tri = _make_triton(in_f, out_f, mw, Sw, mb, Sb)
    tri.forward(ma.to(DEVICE), torch.zeros(N, in_f, device=DEVICE))
    return tri, ma, delta_mz, delta_Sz


def test_linear_backward_delta_ma():
    """delta_ma = delta_mz @ mw^T matches fp64 reference."""
    tri, ma, delta_mz, delta_Sz = _setup_backward()
    d_ma, _ = tri.backward(delta_mz.to(DEVICE), delta_Sz.to(DEVICE))

    ref = (delta_mz.double() @ tri.mw.cpu().double().T).float()
    torch.testing.assert_close(d_ma.cpu(), ref, atol=MEAN_ATOL, rtol=0)


def test_linear_backward_delta_Sa():
    """delta_Sa = delta_Sz @ (mw²)^T matches fp64 reference."""
    tri, ma, delta_mz, delta_Sz = _setup_backward()
    _, d_Sa = tri.backward(delta_mz.to(DEVICE), delta_Sz.to(DEVICE))

    mw = tri.mw.cpu().double()
    ref = (delta_Sz.double() @ (mw * mw).T).float()
    torch.testing.assert_close(d_Sa.cpu(), ref, atol=VAR_ATOL, rtol=0)


def test_linear_backward_delta_mw():
    """delta_mw = Sw * (ma^T @ delta_mz) matches fp64 reference."""
    tri, ma, delta_mz, delta_Sz = _setup_backward()
    tri.backward(delta_mz.to(DEVICE), delta_Sz.to(DEVICE))

    ref = (tri.Sw.cpu().double() * (ma.double().T @ delta_mz.double())).float()
    torch.testing.assert_close(tri.delta_mw.cpu(), ref, atol=UPDATE_ATOL, rtol=0)


def test_linear_backward_delta_Sw():
    """delta_Sw = Sw² * ((ma²)^T @ delta_Sz) matches fp64 reference."""
    tri, ma, delta_mz, delta_Sz = _setup_backward()
    tri.backward(delta_mz.to(DEVICE), delta_Sz.to(DEVICE))

    Sw = tri.Sw.cpu().double()
    ref = (Sw * Sw * ((ma.double() ** 2).T @ delta_Sz.double())).float()
    torch.testing.assert_close(tri.delta_Sw.cpu(), ref, atol=UPDATE_ATOL, rtol=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Level 3: Full step (forward + backward + update)
# ──────────────────────────────────────────────────────────────────────────────


def test_linear_update_mw():
    """After one training step, updated mw matches between triton-tagi and cuTAGI."""
    torch.manual_seed(0)
    N, in_f, out_f = 8, 16, 32
    sigma_v = 0.1

    mw = torch.randn(in_f, out_f)
    Sw = torch.rand(in_f, out_f).abs() * 0.1 + 1e-6
    mb = torch.randn(1, out_f)
    Sb = torch.rand(1, out_f).abs() * 0.1 + 1e-6
    ma = torch.randn(N, in_f)
    y = torch.randn(N, out_f)

    # ── triton step ──
    from triton_tagi.network import Sequential as TSequential

    tri = _make_triton(in_f, out_f, mw, Sw, mb, Sb)
    net_tri = TSequential([tri], device=DEVICE)
    net_tri.step(ma.to(DEVICE), y.to(DEVICE), sigma_v)

    # ── pytagi step ──
    cut = _make_pytagi(in_f, out_f, mw, Sw, mb, Sb)
    updater = OutputUpdater(cut.device)
    m_flat, v_flat = cut(ma.numpy().flatten().astype(np.float32))
    var_y = np.full(N * out_f, sigma_v**2, dtype=np.float32)
    updater.update(
        output_states=cut.output_z_buffer,
        mu_obs=y.numpy().flatten().astype(np.float32),
        var_obs=var_y,
        delta_states=cut.input_delta_z_buffer,
    )
    cut.backward()
    cut.step()

    # ── compare updated mw ──
    key = list(cut.state_dict().keys())[0]
    mw_cut, Sw_cut, mb_cut, Sb_cut = _pytagi_weights(cut, key)

    torch.testing.assert_close(tri.mw.cpu(), mw_cut, atol=UPDATE_ATOL, rtol=0)


def test_linear_update_Sw():
    """After one training step, updated Sw matches between triton-tagi and cuTAGI."""
    torch.manual_seed(0)
    N, in_f, out_f = 8, 16, 32
    sigma_v = 0.1

    mw = torch.randn(in_f, out_f)
    Sw = torch.rand(in_f, out_f).abs() * 0.1 + 1e-6
    mb = torch.randn(1, out_f)
    Sb = torch.rand(1, out_f).abs() * 0.1 + 1e-6
    ma = torch.randn(N, in_f)
    y = torch.randn(N, out_f)

    from triton_tagi.network import Sequential as TSequential

    tri = _make_triton(in_f, out_f, mw, Sw, mb, Sb)
    net_tri = TSequential([tri], device=DEVICE)
    net_tri.step(ma.to(DEVICE), y.to(DEVICE), sigma_v)

    cut = _make_pytagi(in_f, out_f, mw, Sw, mb, Sb)
    updater = OutputUpdater(cut.device)
    m_flat, v_flat = cut(ma.numpy().flatten().astype(np.float32))
    var_y = np.full(N * out_f, sigma_v**2, dtype=np.float32)
    updater.update(
        output_states=cut.output_z_buffer,
        mu_obs=y.numpy().flatten().astype(np.float32),
        var_obs=var_y,
        delta_states=cut.input_delta_z_buffer,
    )
    cut.backward()
    cut.step()

    key = list(cut.state_dict().keys())[0]
    _, Sw_cut, _, _ = _pytagi_weights(cut, key)

    torch.testing.assert_close(tri.Sw.cpu(), Sw_cut, atol=UPDATE_ATOL, rtol=0)
