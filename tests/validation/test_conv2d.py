"""Validation tests: triton-tagi Conv2D against cuTAGI (pytagi).

Conv2D reduces to the same im2col + fused-matmul pipeline as Linear, so the
same three validation levels apply.

Level 1 — Forward: identical weights produce matching (mz, Sz).
Level 2 — Backward: given identical output deltas, input deltas and parameter
           deltas match the fp64 analytical reference.
Level 3 — Update: one full step produces matching updated mw.

Weight layout translation (same as Linear):
    triton  mw : (K, C_out)  where K = C_in * kH * kW
    pytagi  mu_w : (C_out * K,) row-major  →  mw.T.flatten()

pytagi Conv2d requires in_width / in_height at construction time.

Run with:
    pytest tests/validation/test_conv2d.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pytagi.nn import Conv2d as PConv2d
from pytagi.nn import Sequential as PSequential

from triton_tagi.layers.conv2d import Conv2D as TConv2D
from triton_tagi.network import Sequential as TSequential

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN_ATOL = 1e-4
VAR_ATOL = 1e-4
UPDATE_ATOL = 1e-4

pytestmark = pytest.mark.cuda


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_triton_conv(C_in, C_out, k, padding, mw, Sw, mb, Sb):
    layer = TConv2D(C_in, C_out, k, padding=padding, device=DEVICE)
    layer.mw = mw.clone().to(DEVICE)
    layer.Sw = Sw.clone().to(DEVICE)
    layer.mb = mb.clone().to(DEVICE)
    layer.Sb = Sb.clone().to(DEVICE)
    return layer


def _make_pytagi_conv(C_in, C_out, k, padding, H, W, mw, Sw, mb, Sb):
    """Build a pytagi Conv2d with the same weights as the triton layer.

    pytagi weight layout: (C_out, K) row-major flat = mw.T.flatten()
    """
    net = PSequential(PConv2d(C_in, C_out, k, padding=padding, in_width=W, in_height=H))
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


def _pytagi_conv_weights(net, key):
    """Extract (mw, Sw, mb, Sb) from pytagi, converting back to triton layout."""
    mu_w_flat, var_w_flat, mu_b_flat, var_b_flat = net.state_dict()[key]
    C_out = len(mu_b_flat)
    K = len(mu_w_flat) // C_out
    mw = torch.tensor(mu_w_flat).reshape(C_out, K).T  # (K, C_out)
    Sw = torch.tensor(var_w_flat).reshape(C_out, K).T
    mb = torch.tensor(mu_b_flat).reshape(1, C_out)
    Sb = torch.tensor(var_b_flat).reshape(1, C_out)
    return mw, Sw, mb, Sb


def _random_conv_params(C_in, C_out, k):
    K = C_in * k * k
    mw = torch.randn(K, C_out)
    Sw = torch.rand(K, C_out).abs() * 0.1 + 1e-6
    mb = torch.randn(1, C_out)
    Sb = torch.rand(1, C_out).abs() * 0.1 + 1e-6
    return mw, Sw, mb, Sb


# ──────────────────────────────────────────────────────────────────────────────
#  Level 1: Forward
# ──────────────────────────────────────────────────────────────────────────────


def test_conv2d_forward_mean():
    """Output means match between triton-tagi and cuTAGI."""
    torch.manual_seed(0)
    N, C_in, H, W, C_out, k = 4, 3, 8, 8, 8, 3
    mw, Sw, mb, Sb = _random_conv_params(C_in, C_out, k)
    ma = torch.randn(N, C_in, H, W)

    tri = _make_triton_conv(C_in, C_out, k, 1, mw, Sw, mb, Sb)
    mz_tri, _ = tri.forward(ma.to(DEVICE), torch.zeros_like(ma.to(DEVICE)))

    cut = _make_pytagi_conv(C_in, C_out, k, 1, H, W, mw, Sw, mb, Sb)
    m_flat, _ = cut(ma.numpy().flatten().astype(np.float32))
    mz_cut = torch.tensor(m_flat).reshape(N, C_out, H, W)

    torch.testing.assert_close(mz_tri.cpu(), mz_cut, atol=MEAN_ATOL, rtol=0)


def test_conv2d_forward_variance():
    """Output variances match between triton-tagi and cuTAGI."""
    torch.manual_seed(1)
    N, C_in, H, W, C_out, k = 4, 3, 8, 8, 8, 3
    mw, Sw, mb, Sb = _random_conv_params(C_in, C_out, k)
    ma = torch.randn(N, C_in, H, W)

    tri = _make_triton_conv(C_in, C_out, k, 1, mw, Sw, mb, Sb)
    _, Sz_tri = tri.forward(ma.to(DEVICE), torch.zeros_like(ma.to(DEVICE)))

    cut = _make_pytagi_conv(C_in, C_out, k, 1, H, W, mw, Sw, mb, Sb)
    _, v_flat = cut(ma.numpy().flatten().astype(np.float32))
    Sz_cut = torch.tensor(v_flat).reshape(N, C_out, H, W)

    torch.testing.assert_close(Sz_tri.cpu(), Sz_cut, atol=VAR_ATOL, rtol=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Level 2: Backward — input delta propagation and parameter deltas
#
#  After im2col, Conv2D backward is identical to Linear backward on the
#  patch matrix.  Reference formulas:
#    delta_patches_ma = delta_mz_flat @ mw^T         (then col2im)
#    delta_patches_Sa = delta_Sz_flat @ (mw²)^T      (then col2im)
#    delta_mw = Sw * (patches_ma^T @ delta_mz_flat)
#    delta_Sw = Sw² * ((patches_ma²)^T @ delta_Sz_flat)
# ──────────────────────────────────────────────────────────────────────────────


def _setup_conv_backward(seed=0):
    torch.manual_seed(seed)
    N, C_in, H, W, C_out, k = 4, 3, 8, 8, 8, 3
    mw, Sw, mb, Sb = _random_conv_params(C_in, C_out, k)
    ma = torch.randn(N, C_in, H, W)
    delta_mz = torch.randn(N, C_out, H, W)
    delta_Sz = torch.rand(N, C_out, H, W).abs() * 0.01

    tri = _make_triton_conv(C_in, C_out, k, 1, mw, Sw, mb, Sb)
    tri.forward(ma.to(DEVICE), torch.zeros_like(ma.to(DEVICE)))
    return tri, ma, delta_mz, delta_Sz


def test_conv2d_backward_delta_ma():
    """delta_ma matches fp64 reference: (delta_mz_flat @ mw^T) folded via F.fold."""
    import torch.nn.functional as F

    tri, ma, delta_mz, delta_Sz = _setup_conv_backward()
    d_ma, _ = tri.backward(delta_mz.to(DEVICE), delta_Sz.to(DEVICE))

    N, C_in, H, W = tri.input_shape
    H_out, W_out = tri.spatial
    K = C_in * tri.kH * tri.kW

    # fp64 reference: patch-level deltas folded back with F.fold
    dmz_flat = delta_mz.permute(0, 2, 3, 1).reshape(-1, tri.C_out).double()
    dp_ma_ref = (dmz_flat @ tri.mw.cpu().double().T).float()  # (N*L, K)

    # F.fold expects (N, K, L); L = H_out * W_out
    dp_ma_nkl = dp_ma_ref.view(N, H_out * W_out, K).permute(0, 2, 1).contiguous()
    d_ma_ref = F.fold(dp_ma_nkl, output_size=(H, W), kernel_size=tri.kH, padding=tri.padding)

    torch.testing.assert_close(d_ma.cpu(), d_ma_ref, atol=MEAN_ATOL, rtol=0)


def test_conv2d_backward_delta_mw():
    """delta_mw = Sw * (patches_ma^T @ delta_mz_flat) matches fp64 reference."""
    tri, ma, delta_mz, delta_Sz = _setup_conv_backward()
    tri.backward(delta_mz.to(DEVICE), delta_Sz.to(DEVICE))

    patches = tri.patches_ma.cpu().double()
    dmz_flat = delta_mz.permute(0, 2, 3, 1).reshape(-1, tri.C_out).double()
    ref = (tri.Sw.cpu().double() * (patches.T @ dmz_flat)).float()

    torch.testing.assert_close(tri.delta_mw.cpu(), ref, atol=UPDATE_ATOL, rtol=0)


def test_conv2d_backward_delta_Sw():
    """delta_Sw = Sw² * ((patches_ma²)^T @ delta_Sz_flat) matches fp64 reference."""
    tri, ma, delta_mz, delta_Sz = _setup_conv_backward()
    tri.backward(delta_mz.to(DEVICE), delta_Sz.to(DEVICE))

    patches = tri.patches_ma.cpu().double()
    dSz_flat = delta_Sz.permute(0, 2, 3, 1).reshape(-1, tri.C_out).double()
    Sw = tri.Sw.cpu().double()
    ref = (Sw * Sw * ((patches**2).T @ dSz_flat)).float()

    torch.testing.assert_close(tri.delta_Sw.cpu(), ref, atol=UPDATE_ATOL, rtol=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Level 3: Full step (forward + backward + update)
#
#  pytagi's Conv2d.backward() segfaults in the installed version, so the
#  end-to-end check uses a Python fp64 reference built from the same update
#  chain: F.unfold → forward → innovation → backward → capped update.
#  Each component is independently validated in Levels 1 and 2; Level 3
#  checks that the Sequential wires them together correctly.
# ──────────────────────────────────────────────────────────────────────────────


def test_conv2d_update_mw():
    """After one step, updated mw matches the TAGI update formula (fp64 reference)."""
    import torch.nn.functional as F

    from triton_tagi.update.parameters import get_cap_factor

    torch.manual_seed(0)
    N, C_in, H, W, C_out, k = 4, 3, 8, 8, 8, 3
    sigma_v = 0.1
    mw, Sw, mb, Sb = _random_conv_params(C_in, C_out, k)
    ma = torch.randn(N, C_in, H, W)
    y = torch.randn(N, C_out, H, W)

    # triton step
    tri = _make_triton_conv(C_in, C_out, k, 1, mw, Sw, mb, Sb)
    net_tri = TSequential([tri], device=DEVICE)
    net_tri.step(ma.to(DEVICE), y.to(DEVICE), sigma_v)

    # fp64 reference using F.unfold (independent of triton im2col)
    # Sequential passes Sa = zeros_like(x), so cross-term Sa@(mw²+Sw) vanishes.
    K = C_in * k * k
    patches = F.unfold(ma, kernel_size=k, padding=1)  # (N, K, L)
    L = patches.shape[2]
    patches = patches.permute(0, 2, 1).reshape(N * L, K).double()  # (N*L, K)

    mw64, Sw64 = mw.double(), Sw.double()
    mb64, Sb64 = mb.double(), Sb.double()

    mz_flat = patches @ mw64 + mb64               # (N*L, C_out)
    Sz_flat = patches ** 2 @ Sw64 + Sb64          # Sa=0 → only patch² @ Sw + Sb

    mz = mz_flat.view(N, H, W, C_out).permute(0, 3, 1, 2)
    Sz = Sz_flat.view(N, H, W, C_out).permute(0, 3, 1, 2)

    dmz = (y.double() - mz) / (Sz + sigma_v ** 2)
    dmz_flat = dmz.permute(0, 2, 3, 1).reshape(N * L, C_out)

    delta_mw = Sw64 * (patches.T @ dmz_flat)

    cap = get_cap_factor(N)
    delta_bar = Sw64.sqrt() / cap
    dmw_capped = torch.sign(delta_mw) * torch.minimum(delta_mw.abs(), delta_bar)
    mw_ref = (mw64 + dmw_capped).float()

    torch.testing.assert_close(tri.mw.cpu(), mw_ref, atol=UPDATE_ATOL, rtol=0)
