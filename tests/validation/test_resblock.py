"""Validation tests: triton-tagi ResBlock against manual sub-layer assembly.

pytagi's ResNetBlock crashes (FPE) in the installed version.  The validation
approach is therefore:

  Level 1 — Forward composition: create a ResBlock and a manual pipeline of
             the same sub-layers with identical weights; both must produce
             the same (mz, Sz).

  Level 2 — Backward composition: given identical output deltas, the delta
             reaching the ResBlock input must equal the delta from manually
             chaining backward through the same sub-layers, summing the
             shortcut branch as ResBlock.backward does.

  Level 3 — Update: after one full step, conv1.mw equals
             mw_init + capped(conv1.delta_mw) — verifies that the update
             was actually applied to the sub-layer and that the capping
             formula is correct.

Architecture validated:
    Identity shortcut   in_ch == out_ch, stride == 1
    Projection shortcut in_ch != out_ch or stride != 1

Run with:
    pytest tests/validation/test_resblock.py -v
"""

from __future__ import annotations

import pytest
import torch

from triton_tagi.layers.batchnorm2d import BatchNorm2D as TBatchNorm2D
from triton_tagi.layers.conv2d import Conv2D as TConv2D
from triton_tagi.layers.relu import ReLU as TReLU
from triton_tagi.layers.resblock import ResBlock as TResBlock
from triton_tagi.network import Sequential as TSequential
from triton_tagi.update.parameters import get_cap_factor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FWD_ATOL = 1e-5
BWD_ATOL = 1e-5
UPD_ATOL = 1e-6

pytestmark = pytest.mark.cuda


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _copy_conv_weights(src: TConv2D, dst: TConv2D) -> None:
    dst.mw = src.mw.clone()
    dst.Sw = src.Sw.clone()
    dst.mb = src.mb.clone()
    dst.Sb = src.Sb.clone()


def _copy_bn_weights(src: TBatchNorm2D, dst: TBatchNorm2D) -> None:
    dst.mw = src.mw.clone()
    dst.Sw = src.Sw.clone()
    dst.mb = src.mb.clone()
    dst.Sb = src.Sb.clone()
    dst.running_mean = src.running_mean.clone()
    dst.running_var = src.running_var.clone()
    dst._is_initialized = src._is_initialized


# ──────────────────────────────────────────────────────────────────────────────
#  Level 1: Forward — identity shortcut
# ──────────────────────────────────────────────────────────────────────────────


def test_resblock_identity_forward_mean():
    """Identity ResBlock forward mean = manual (Conv→ReLU→BN)² + skip."""
    torch.manual_seed(0)
    N, C, H, W = 4, 8, 8, 8
    ma = torch.randn(N, C, H, W, device=DEVICE)
    Sa = torch.rand(N, C, H, W, device=DEVICE).abs() * 0.1 + 1e-4

    blk = TResBlock(C, C, stride=1, device=DEVICE)
    mz_blk, _ = blk.forward(ma, Sa)

    # Manual pipeline: identical weights
    c1 = TConv2D(C, C, 3, padding=1, device=DEVICE)
    r1 = TReLU()
    b1 = TBatchNorm2D(C, device=DEVICE, preserve_var=False)
    c2 = TConv2D(C, C, 3, padding=1, device=DEVICE)
    r2 = TReLU()
    b2 = TBatchNorm2D(C, device=DEVICE, preserve_var=False)

    _copy_conv_weights(blk.conv1, c1)
    _copy_bn_weights(blk.bn1, b1)
    _copy_conv_weights(blk.conv2, c2)
    _copy_bn_weights(blk.bn2, b2)

    mz_m, Sz_m = c1.forward(ma, Sa)
    mz_m, Sz_m = r1.forward(mz_m, Sz_m)
    mz_m, Sz_m = b1.forward(mz_m, Sz_m)
    mz_m, Sz_m = c2.forward(mz_m, Sz_m)
    mz_m, Sz_m = r2.forward(mz_m, Sz_m)
    mz_m, Sz_m = b2.forward(mz_m, Sz_m)
    mz_m = mz_m + ma          # identity shortcut
    Sz_m = Sz_m + Sa

    torch.testing.assert_close(mz_blk.cpu(), mz_m.cpu(), atol=FWD_ATOL, rtol=0)


def test_resblock_identity_forward_variance():
    """Identity ResBlock forward variance = manual pipeline + skip."""
    torch.manual_seed(1)
    N, C, H, W = 4, 8, 8, 8
    ma = torch.randn(N, C, H, W, device=DEVICE)
    Sa = torch.rand(N, C, H, W, device=DEVICE).abs() * 0.1 + 1e-4

    blk = TResBlock(C, C, stride=1, device=DEVICE)
    _, Sz_blk = blk.forward(ma, Sa)

    c1 = TConv2D(C, C, 3, padding=1, device=DEVICE)
    r1 = TReLU()
    b1 = TBatchNorm2D(C, device=DEVICE, preserve_var=False)
    c2 = TConv2D(C, C, 3, padding=1, device=DEVICE)
    r2 = TReLU()
    b2 = TBatchNorm2D(C, device=DEVICE, preserve_var=False)

    _copy_conv_weights(blk.conv1, c1)
    _copy_bn_weights(blk.bn1, b1)
    _copy_conv_weights(blk.conv2, c2)
    _copy_bn_weights(blk.bn2, b2)

    mz_m, Sz_m = c1.forward(ma, Sa)
    mz_m, Sz_m = r1.forward(mz_m, Sz_m)
    mz_m, Sz_m = b1.forward(mz_m, Sz_m)
    mz_m, Sz_m = c2.forward(mz_m, Sz_m)
    mz_m, Sz_m = r2.forward(mz_m, Sz_m)
    mz_m, Sz_m = b2.forward(mz_m, Sz_m)
    Sz_m = Sz_m + Sa

    torch.testing.assert_close(Sz_blk.cpu(), Sz_m.cpu(), atol=FWD_ATOL, rtol=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Level 1: Forward — projection shortcut
# ──────────────────────────────────────────────────────────────────────────────


def test_resblock_projection_forward_mean():
    """Projection ResBlock forward mean = manual main + projection path + merge."""
    torch.manual_seed(0)
    N, C_in, C_out, H, W = 4, 4, 8, 8, 8
    ma = torch.randn(N, C_in, H, W, device=DEVICE)
    Sa = torch.rand(N, C_in, H, W, device=DEVICE).abs() * 0.1 + 1e-4

    blk = TResBlock(C_in, C_out, stride=2, device=DEVICE)
    mz_blk, _ = blk.forward(ma, Sa)

    # Main path (stride=2 on first conv)
    c1 = TConv2D(C_in, C_out, 3, stride=2, padding=1, padding_type=2, device=DEVICE)
    r1 = TReLU()
    b1 = TBatchNorm2D(C_out, device=DEVICE, preserve_var=False)
    c2 = TConv2D(C_out, C_out, 3, stride=1, padding=1, device=DEVICE)
    r2 = TReLU()
    b2 = TBatchNorm2D(C_out, device=DEVICE, preserve_var=False)
    # Projection shortcut (k=2, stride=2)
    pc = TConv2D(C_in, C_out, 2, stride=2, padding=0, device=DEVICE)
    pr = TReLU()
    pb = TBatchNorm2D(C_out, device=DEVICE, preserve_var=False)

    _copy_conv_weights(blk.conv1, c1)
    _copy_bn_weights(blk.bn1, b1)
    _copy_conv_weights(blk.conv2, c2)
    _copy_bn_weights(blk.bn2, b2)
    _copy_conv_weights(blk.proj_conv, pc)
    _copy_bn_weights(blk.proj_bn, pb)

    # Main path forward
    mz_m, Sz_m = c1.forward(ma, Sa)
    mz_m, Sz_m = r1.forward(mz_m, Sz_m)
    mz_m, Sz_m = b1.forward(mz_m, Sz_m)
    mz_m, Sz_m = c2.forward(mz_m, Sz_m)
    mz_m, Sz_m = r2.forward(mz_m, Sz_m)
    mz_m, Sz_m = b2.forward(mz_m, Sz_m)

    # Projection path forward
    mz_p, Sz_p = pc.forward(ma, Sa)
    mz_p, Sz_p = pr.forward(mz_p, Sz_p)
    mz_p, Sz_p = pb.forward(mz_p, Sz_p)

    # Merge
    mz_ref = mz_m + mz_p

    torch.testing.assert_close(mz_blk.cpu(), mz_ref.cpu(), atol=FWD_ATOL, rtol=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Level 2: Backward — identity shortcut
# ──────────────────────────────────────────────────────────────────────────────


def test_resblock_identity_backward_delta_ma():
    """Identity ResBlock backward: delta_in = main_backward + delta_out (skip)."""
    torch.manual_seed(0)
    N, C, H, W = 4, 8, 8, 8
    ma = torch.randn(N, C, H, W, device=DEVICE)
    Sa = torch.rand(N, C, H, W, device=DEVICE).abs() * 0.1 + 1e-4
    delta_mz = torch.randn(N, C, H, W, device=DEVICE)
    delta_Sz = torch.rand(N, C, H, W, device=DEVICE).abs() * 0.01

    blk = TResBlock(C, C, stride=1, device=DEVICE)
    blk.forward(ma, Sa)
    d_ma, _ = blk.backward(delta_mz, delta_Sz)

    # Manual backward using the same cached state
    # Run forward through copies to get cached BN/ReLU state
    c1 = TConv2D(C, C, 3, padding=1, device=DEVICE)
    r1 = TReLU()
    b1 = TBatchNorm2D(C, device=DEVICE, preserve_var=False)
    c2 = TConv2D(C, C, 3, padding=1, device=DEVICE)
    r2 = TReLU()
    b2 = TBatchNorm2D(C, device=DEVICE, preserve_var=False)

    _copy_conv_weights(blk.conv1, c1)
    _copy_bn_weights(blk.bn1, b1)
    _copy_conv_weights(blk.conv2, c2)
    _copy_bn_weights(blk.bn2, b2)

    mz_m, Sz_m = c1.forward(ma, Sa)
    mz_m, Sz_m = r1.forward(mz_m, Sz_m)
    mz_m, Sz_m = b1.forward(mz_m, Sz_m)
    mz_m, Sz_m = c2.forward(mz_m, Sz_m)
    mz_m, Sz_m = r2.forward(mz_m, Sz_m)
    b2.forward(mz_m, Sz_m)

    # Backward through main path (reversed) with the same incoming deltas
    dm, dv = delta_mz.clone(), delta_Sz.clone()
    dm, dv = b2.backward(dm, dv)
    dm, dv = r2.backward(dm, dv)
    dm, dv = c2.backward(dm, dv)
    dm, dv = b1.backward(dm, dv)
    dm, dv = r1.backward(dm, dv)
    dm, dv = c1.backward(dm, dv)

    # Identity shortcut: add the original delta (Jacobian = 1)
    dm = dm + delta_mz
    dv = dv + delta_Sz

    torch.testing.assert_close(d_ma.cpu(), dm.cpu(), atol=BWD_ATOL, rtol=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Level 3: Update — verify delta_mw applied correctly to conv1
# ──────────────────────────────────────────────────────────────────────────────


def test_resblock_identity_update_conv1_mw():
    """After one step, conv1.mw = mw_init + capped(conv1.delta_mw)."""
    torch.manual_seed(0)
    N, C, H, W = 4, 8, 8, 8
    sigma_v = 0.1
    ma = torch.randn(N, C, H, W, device=DEVICE)
    y = torch.randn(N, C, H, W, device=DEVICE)

    blk = TResBlock(C, C, stride=1, device=DEVICE)
    mw_init = blk.conv1.mw.cpu().clone()
    Sw_init = blk.conv1.Sw.cpu().clone()

    net = TSequential([blk], device=DEVICE)
    net.step(ma, y, sigma_v)

    # After step: delta_mw is stored, mw is updated
    delta_mw = blk.conv1.delta_mw.cpu()
    cap = get_cap_factor(N)
    delta_bar = torch.sqrt(torch.clamp(Sw_init, min=1e-10)) / cap
    dmw_capped = torch.sign(delta_mw) * torch.minimum(delta_mw.abs(), delta_bar)
    mw_expected = mw_init + dmw_capped

    torch.testing.assert_close(blk.conv1.mw.cpu(), mw_expected, atol=UPD_ATOL, rtol=0)


def test_resblock_projection_update_conv1_mw():
    """After one step with projection shortcut, conv1.mw = mw_init + capped(delta_mw)."""
    torch.manual_seed(0)
    N, C_in, C_out, H, W = 4, 4, 8, 8, 8
    sigma_v = 0.1
    ma = torch.randn(N, C_in, H, W, device=DEVICE)
    y = torch.randn(N, C_out, H // 2, W // 2, device=DEVICE)

    blk = TResBlock(C_in, C_out, stride=2, device=DEVICE)
    mw_init = blk.conv1.mw.cpu().clone()
    Sw_init = blk.conv1.Sw.cpu().clone()

    net = TSequential([blk], device=DEVICE)
    net.step(ma, y, sigma_v)

    delta_mw = blk.conv1.delta_mw.cpu()
    cap = get_cap_factor(N)
    delta_bar = torch.sqrt(torch.clamp(Sw_init, min=1e-10)) / cap
    dmw_capped = torch.sign(delta_mw) * torch.minimum(delta_mw.abs(), delta_bar)
    mw_expected = mw_init + dmw_capped

    torch.testing.assert_close(blk.conv1.mw.cpu(), mw_expected, atol=UPD_ATOL, rtol=0)
