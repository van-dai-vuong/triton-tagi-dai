"""Validation tests: triton-tagi initialization against cuTAGI (pytagi).

Weight means (mw, mb) are drawn from N(0, scale) with independent RNGs, so
element-wise comparison is not possible.  We validate the deterministic
quantities instead:

  - Sw / Sb values (constant per layer: (scale * gain)^2)
  - scale formula (He: sqrt(1/fan_in); Xavier: sqrt(2/(fan_in+fan_out)))
  - std of mw sample matches within statistical tolerance

Run with:
    pytest tests/validation/test_init.py -v
"""

from __future__ import annotations

import math

import pytest
import torch
from pytagi.nn import Linear as PLinear
from pytagi.nn import Sequential as PSequential

from triton_tagi.layers.batchnorm2d import BatchNorm2D as TBatchNorm2D
from triton_tagi.layers.conv2d import Conv2D as TConv2D
from triton_tagi.layers.linear import Linear as TLinear

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pytestmark = pytest.mark.cuda


# ──────────────────────────────────────────────────────────────────────────────
#  He initialization
# ──────────────────────────────────────────────────────────────────────────────


def test_he_init_Sw_matches_cutagi():
    """Sw after default He init equals cuTAGI's var_w = 1/fan_in."""
    in_f, out_f = 64, 128
    tri = TLinear(in_f, out_f, device=DEVICE)

    expected_Sw = 1.0 / in_f  # cuTAGI: scale=sqrt(1/fan_in), Sw=scale^2
    assert tri.Sw.shape == (in_f, out_f)
    torch.testing.assert_close(
        tri.Sw.cpu(),
        torch.full((in_f, out_f), expected_Sw),
        atol=0.0,
        rtol=0.0,
        msg=f"Sw={tri.Sw[0,0].item():.6f}, expected {expected_Sw:.6f}",
    )


def test_he_init_Sb_matches_cutagi():
    """Sb after default He init equals cuTAGI's var_b = 1/fan_in."""
    in_f, out_f = 64, 128
    tri = TLinear(in_f, out_f, device=DEVICE)

    expected_Sb = 1.0 / in_f
    torch.testing.assert_close(
        tri.Sb.cpu(),
        torch.full((1, out_f), expected_Sb),
        atol=0.0,
        rtol=0.0,
    )


def test_he_init_Sw_matches_pytagi():
    """Sw from triton-tagi exactly equals var_w from a fresh pytagi Linear."""
    in_f, out_f = 32, 64
    tri = TLinear(in_f, out_f, device=DEVICE)

    net = PSequential(PLinear(in_f, out_f))
    net.preinit_layer()
    key = list(net.state_dict().keys())[0]
    _, var_w_flat, _, var_b_flat = net.state_dict()[key]

    expected_Sw = var_w_flat[0]  # all equal for He init
    expected_Sb = var_b_flat[0]

    # atol=1e-7 tolerates the float32→Python-float round-trip in pytagi's C++ backend
    assert abs(tri.Sw[0, 0].item() - expected_Sw) < 1e-7, (
        f"Sw mismatch: triton={tri.Sw[0,0].item()}, pytagi={expected_Sw}"
    )
    assert abs(tri.Sb[0, 0].item() - expected_Sb) < 1e-7, (
        f"Sb mismatch: triton={tri.Sb[0,0].item()}, pytagi={expected_Sb}"
    )


def test_he_init_mw_scale():
    """std(mw) is statistically consistent with N(0, sqrt(1/fan_in))."""
    torch.manual_seed(0)
    in_f, out_f = 256, 512  # large enough for a tight empirical std
    tri = TLinear(in_f, out_f, device=DEVICE)

    expected_std = math.sqrt(1.0 / in_f)
    empirical_std = tri.mw.std().item()
    # allow ±10% relative tolerance for a 256×512 sample
    assert abs(empirical_std - expected_std) / expected_std < 0.10, (
        f"mw std={empirical_std:.5f}, expected≈{expected_std:.5f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Xavier initialization
# ──────────────────────────────────────────────────────────────────────────────


def test_xavier_init_Sw():
    """Xavier init: Sw = 2/(fan_in+fan_out) — matches cuTAGI."""
    in_f, out_f = 32, 64
    tri = TLinear(in_f, out_f, device=DEVICE, init_method="Xavier")

    expected_Sw = 2.0 / (in_f + out_f)
    torch.testing.assert_close(
        tri.Sw.cpu(),
        torch.full((in_f, out_f), expected_Sw),
        atol=0.0,
        rtol=0.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Conv2D initialization
# ──────────────────────────────────────────────────────────────────────────────


def test_conv2d_he_init_Sw():
    """Conv2D He init: Sw = 1/fan_in where fan_in = C_in*kH*kW — matches cuTAGI."""
    C_in, C_out, k = 3, 16, 3
    fan_in = C_in * k * k
    tri = TConv2D(C_in, C_out, k, device=DEVICE)

    expected_Sw = 1.0 / fan_in
    assert abs(tri.Sw[0, 0].item() - expected_Sw) / expected_Sw < 1e-6


def test_conv2d_he_init_matches_pytagi():
    """Conv2D Sw matches pytagi's var_w to float32 precision."""
    from pytagi.nn import Conv2d as PConv2d
    from pytagi.nn import Sequential as PSequential

    C_in, C_out, k, H, W = 3, 16, 3, 8, 8
    tri = TConv2D(C_in, C_out, k, device=DEVICE)

    net = PSequential(PConv2d(C_in, C_out, k, in_width=W, in_height=H))
    net.preinit_layer()
    key = list(net.state_dict().keys())[0]
    _, var_w_flat, _, var_b_flat = net.state_dict()[key]

    assert abs(tri.Sw[0, 0].item() - var_w_flat[0]) < 1e-7
    assert abs(tri.Sb[0, 0].item() - var_b_flat[0]) < 1e-7


# ──────────────────────────────────────────────────────────────────────────────
#  BatchNorm2D initialization
# ──────────────────────────────────────────────────────────────────────────────


def test_batchnorm_init_Sw_matches_cutagi():
    """BN Sw = 2/(C+C) = 1/C — matches cuTAGI init_weight_bias_norm."""
    C = 32
    bn = TBatchNorm2D(C, device=DEVICE)

    expected = 2.0 / (C + C)
    assert abs(bn.Sw[0].item() - expected) / expected < 1e-6
    assert abs(bn.Sb[0].item() - expected) / expected < 1e-6


def test_batchnorm_init_means():
    """BN mg (gamma) = 1.0, mb (beta) = 0.0 — matches cuTAGI."""
    C = 16
    bn = TBatchNorm2D(C, device=DEVICE)
    assert torch.all(bn.mw == 1.0)
    assert torch.all(bn.mb == 0.0)


def test_batchnorm_init_Sw_matches_pytagi():
    """BN Sw matches pytagi's Sg to float32 precision."""
    from pytagi.nn import BatchNorm2d as PBatchNorm2d
    from pytagi.nn import Sequential as PSequential

    C = 16
    bn = TBatchNorm2D(C, device=DEVICE)

    net = PSequential(PBatchNorm2d(C))
    net.preinit_layer()
    key = list(net.state_dict().keys())[0]
    _, Sg_flat, _, Sb_flat = net.state_dict()[key]

    assert abs(bn.Sw[0].item() - Sg_flat[0]) < 1e-7
    assert abs(bn.Sb[0].item() - Sb_flat[0]) < 1e-7
