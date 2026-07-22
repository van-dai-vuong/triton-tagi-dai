"""Validation tests: triton-tagi MLP (Linear + ReLU) against cuTAGI (pytagi).

Tests a 2-layer MLP — Linear → ReLU → Linear — which exercises both the Linear
and the ReLU (MixtureReLU in cuTAGI) layers simultaneously. This catches
cross-layer interaction bugs that isolated unit tests miss.

Level 1 — Forward: identical weights produce matching (m, v) at the output.
Level 2 — Backward: given identical output deltas, delta_ma/delta_Sa
           propagating all the way back to the network input match fp64.
Level 3 — Update: one full step produces matching updated weights in both layers.

Run with:
    pytest tests/validation/test_mlp.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pytagi.nn import Linear as PLinear
from pytagi.nn import MixtureReLU, OutputUpdater
from pytagi.nn import Sequential as PSequential

from triton_tagi.layers.linear import Linear as TLinear
from triton_tagi.layers.relu import ReLU as TReLU
from triton_tagi.network import Sequential as TSequential

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Tolerances after tf32 fix: tl.dot uses allow_tf32=False and torch.matmul
# has tf32 disabled at import; triton now matches cuTAGI to ~1e-5 across layers.
MEAN_ATOL = 1e-4  # MLP mean: ~8e-6 max after fix
VAR_ATOL = 1e-3  # MLP variance: ~3e-5 max (small ReLU moment cascade remains)
UPDATE_ATOL = 1e-4  # weight update: Sz nearly identical → tighter update match

pytestmark = pytest.mark.cuda


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _build_triton_mlp(in_f, hidden, out_f, params):
    """Build a triton-tagi Sequential(Linear, ReLU, Linear) with given weights."""
    mw0, Sw0, mb0, Sb0, mw1, Sw1, mb1, Sb1 = params
    l0 = TLinear(in_f, hidden, device=DEVICE)
    l0.mw, l0.Sw, l0.mb, l0.Sb = (
        mw0.clone().to(DEVICE),
        Sw0.clone().to(DEVICE),
        mb0.clone().to(DEVICE),
        Sb0.clone().to(DEVICE),
    )
    l1 = TLinear(hidden, out_f, device=DEVICE)
    l1.mw, l1.Sw, l1.mb, l1.Sb = (
        mw1.clone().to(DEVICE),
        Sw1.clone().to(DEVICE),
        mb1.clone().to(DEVICE),
        Sb1.clone().to(DEVICE),
    )
    return TSequential([l0, TReLU(), l1], device=DEVICE)


def _build_pytagi_mlp(in_f, hidden, out_f, params):
    """Build a pytagi Sequential(Linear, MixtureReLU, Linear) with the same weights.

    pytagi key naming: Linear layers get keys "Linear.0" and "Linear.2"
    (MixtureReLU is at index 1 but has no parameters).
    """
    mw0, Sw0, mb0, Sb0, mw1, Sw1, mb1, Sb1 = params
    net = PSequential(PLinear(in_f, hidden), MixtureReLU(), PLinear(hidden, out_f))
    net.preinit_layer()

    def _flat(mw, Sw, mb, Sb):
        return (
            mw.T.cpu().numpy().flatten().tolist(),
            Sw.T.cpu().numpy().flatten().tolist(),
            mb.squeeze().cpu().numpy().tolist(),
            Sb.squeeze().cpu().numpy().tolist(),
        )

    sd = net.state_dict()
    keys = sorted(sd.keys())  # ["Linear.0", "Linear.2"]
    net.load_state_dict({keys[0]: _flat(mw0, Sw0, mb0, Sb0), keys[1]: _flat(mw1, Sw1, mb1, Sb1)})
    return net


def _pytagi_linear_weights(net, key):
    """Extract (mw, Sw, mb, Sb) from a pytagi net for a given Linear key.

    Converts from pytagi's (out, in) flat layout to triton's (in, out) shape.
    """
    mu_w_flat, var_w_flat, mu_b_flat, var_b_flat = net.state_dict()[key]
    out_f = len(mu_b_flat)
    in_f = len(mu_w_flat) // out_f
    mw = torch.tensor(mu_w_flat).reshape(out_f, in_f).T
    Sw = torch.tensor(var_w_flat).reshape(out_f, in_f).T
    mb = torch.tensor(mu_b_flat).reshape(1, out_f)
    Sb = torch.tensor(var_b_flat).reshape(1, out_f)
    return mw, Sw, mb, Sb


def _random_params(in_f, hidden, out_f):
    mw0 = torch.randn(in_f, hidden)
    Sw0 = torch.rand(in_f, hidden).abs() * 0.1 + 1e-6
    mb0 = torch.randn(1, hidden)
    Sb0 = torch.rand(1, hidden).abs() * 0.1 + 1e-6
    mw1 = torch.randn(hidden, out_f)
    Sw1 = torch.rand(hidden, out_f).abs() * 0.1 + 1e-6
    mb1 = torch.randn(1, out_f)
    Sb1 = torch.rand(1, out_f).abs() * 0.1 + 1e-6
    return mw0, Sw0, mb0, Sb0, mw1, Sw1, mb1, Sb1


# ──────────────────────────────────────────────────────────────────────────────
#  Level 1: Forward
# ──────────────────────────────────────────────────────────────────────────────


def test_mlp_forward_mean():
    """Output means of a 2-layer MLP match between triton-tagi and cuTAGI."""
    torch.manual_seed(0)
    N, in_f, hidden, out_f = 8, 16, 32, 8

    params = _random_params(in_f, hidden, out_f)
    ma = torch.randn(N, in_f)

    tri = _build_triton_mlp(in_f, hidden, out_f, params)
    m_tri, _ = tri.forward(ma.to(DEVICE))

    cut = _build_pytagi_mlp(in_f, hidden, out_f, params)
    m_flat, _ = cut(ma.numpy().flatten().astype(np.float32))
    m_cut = torch.tensor(m_flat).reshape(N, out_f)

    torch.testing.assert_close(m_tri.cpu(), m_cut, atol=MEAN_ATOL, rtol=0)


def test_mlp_forward_variance():
    """Output variances of a 2-layer MLP match between triton-tagi and cuTAGI."""
    torch.manual_seed(1)
    N, in_f, hidden, out_f = 8, 16, 32, 8

    params = _random_params(in_f, hidden, out_f)
    ma = torch.randn(N, in_f)

    tri = _build_triton_mlp(in_f, hidden, out_f, params)
    _, Sz_tri = tri.forward(ma.to(DEVICE))

    cut = _build_pytagi_mlp(in_f, hidden, out_f, params)
    _, v_flat = cut(ma.numpy().flatten().astype(np.float32))
    Sz_cut = torch.tensor(v_flat).reshape(N, out_f)

    torch.testing.assert_close(Sz_tri.cpu(), Sz_cut, atol=VAR_ATOL, rtol=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Level 2: Backward — delta propagation through the full MLP
#
#  The MLP backward chain is:
#    delta at output
#      → Linear1.backward  →  delta_mz_h = d_out @ mw1.T
#                              delta_Sz_h = d_Sout @ mw1².T
#      → ReLU.backward     →  delta_mz_p = delta_mz_h * J
#                              delta_Sz_p = delta_Sz_h * J²
#      → Linear0.backward  →  delta_ma_in = delta_mz_p @ mw0.T
#                              delta_Sa_in = delta_Sz_p @ mw0².T
#
#  Reference: fp64 computation of the same chain, using J from triton's
#  ReLU forward (fp32, accurate to ~1e-7 vs reference).
# ──────────────────────────────────────────────────────────────────────────────


def _mlp_backward_ref(ma_in, delta_mz_out, delta_Sz_out, mw0, mw1, J):
    """fp64 reference for the full MLP backward."""
    # Layer 1 backward
    delta_mz_h = delta_mz_out.double() @ mw1.double().T
    delta_Sz_h = delta_Sz_out.double() @ (mw1.double() ** 2).T

    # ReLU backward (J already in fp32 from triton; promote for the formula)
    J64 = J.double()
    delta_mz_p = delta_mz_h * J64
    delta_Sz_p = delta_Sz_h * J64 * J64

    # Layer 0 backward
    delta_ma_in = delta_mz_p @ mw0.double().T
    delta_Sa_in = delta_Sz_p @ (mw0.double() ** 2).T

    return delta_ma_in.float(), delta_Sa_in.float()


def test_mlp_backward_delta_ma():
    """delta_ma at network input matches fp64 backward reference."""
    torch.manual_seed(0)
    N, in_f, hidden, out_f = 8, 16, 32, 8
    params = _random_params(in_f, hidden, out_f)
    mw0, _, _, _, mw1, _, _, _ = params
    ma = torch.randn(N, in_f)
    delta_mz_out = torch.randn(N, out_f)
    delta_Sz_out = torch.rand(N, out_f).abs() * 0.01

    tri = _build_triton_mlp(in_f, hidden, out_f, params)
    tri.forward(ma.to(DEVICE))

    # Run backward manually to capture the full input delta
    d_mu, d_var = delta_mz_out.to(DEVICE), delta_Sz_out.to(DEVICE)
    for layer in reversed(tri.layers):
        d_mu, d_var = layer.backward(d_mu, d_var)

    # fp64 reference using J from the ReLU forward
    relu = tri.layers[1]
    ref_ma, _ = _mlp_backward_ref(ma, delta_mz_out, delta_Sz_out, mw0, mw1, relu.J.cpu())

    torch.testing.assert_close(d_mu.cpu(), ref_ma, atol=MEAN_ATOL, rtol=0)


def test_mlp_backward_delta_Sa():
    """delta_Sa at network input matches fp64 backward reference."""
    torch.manual_seed(1)
    N, in_f, hidden, out_f = 8, 16, 32, 8
    params = _random_params(in_f, hidden, out_f)
    mw0, _, _, _, mw1, _, _, _ = params
    ma = torch.randn(N, in_f)
    delta_mz_out = torch.randn(N, out_f)
    delta_Sz_out = torch.rand(N, out_f).abs() * 0.01

    tri = _build_triton_mlp(in_f, hidden, out_f, params)
    tri.forward(ma.to(DEVICE))

    d_mu, d_var = delta_mz_out.to(DEVICE), delta_Sz_out.to(DEVICE)
    for layer in reversed(tri.layers):
        d_mu, d_var = layer.backward(d_mu, d_var)

    relu = tri.layers[1]
    _, ref_Sa = _mlp_backward_ref(ma, delta_mz_out, delta_Sz_out, mw0, mw1, relu.J.cpu())

    torch.testing.assert_close(d_var.cpu(), ref_Sa, atol=VAR_ATOL, rtol=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Level 3: Full step
# ──────────────────────────────────────────────────────────────────────────────


def _run_pytagi_step(cut, ma, y, sigma_v):
    N, out_f = y.shape
    updater = OutputUpdater(cut.device)
    cut(ma.numpy().flatten().astype(np.float32))
    var_y = np.full(N * out_f, sigma_v**2, dtype=np.float32)
    updater.update(
        output_states=cut.output_z_buffer,
        mu_obs=y.numpy().flatten().astype(np.float32),
        var_obs=var_y,
        delta_states=cut.input_delta_z_buffer,
    )
    cut.backward()
    cut.step()


def test_mlp_update_layer0_mw():
    """After one step, updated mw for the first Linear matches cuTAGI."""
    torch.manual_seed(0)
    N, in_f, hidden, out_f = 8, 16, 32, 8
    sigma_v = 0.1

    params = _random_params(in_f, hidden, out_f)
    ma = torch.randn(N, in_f)
    y = torch.randn(N, out_f)

    tri = _build_triton_mlp(in_f, hidden, out_f, params)
    tri.step(ma.to(DEVICE), y.to(DEVICE), sigma_v)

    cut = _build_pytagi_mlp(in_f, hidden, out_f, params)
    _run_pytagi_step(cut, ma, y, sigma_v)

    keys = sorted(cut.state_dict().keys())
    mw_cut, _, _, _ = _pytagi_linear_weights(cut, keys[0])

    torch.testing.assert_close(tri.layers[0].mw.cpu(), mw_cut, atol=UPDATE_ATOL, rtol=0)


def test_mlp_update_layer1_mw():
    """After one step, updated mw for the second Linear matches cuTAGI."""
    torch.manual_seed(0)
    N, in_f, hidden, out_f = 8, 16, 32, 8
    sigma_v = 0.1

    params = _random_params(in_f, hidden, out_f)
    ma = torch.randn(N, in_f)
    y = torch.randn(N, out_f)

    tri = _build_triton_mlp(in_f, hidden, out_f, params)
    tri.step(ma.to(DEVICE), y.to(DEVICE), sigma_v)

    cut = _build_pytagi_mlp(in_f, hidden, out_f, params)
    _run_pytagi_step(cut, ma, y, sigma_v)

    keys = sorted(cut.state_dict().keys())
    mw_cut, _, _, _ = _pytagi_linear_weights(cut, keys[1])

    torch.testing.assert_close(tri.layers[2].mw.cpu(), mw_cut, atol=UPDATE_ATOL, rtol=0)
