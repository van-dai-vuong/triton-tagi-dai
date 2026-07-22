"""Unit tests for triton_tagi.layers.layernorm.LayerNorm (1-D / MLP mode).

No cuTAGI dependency — purely tests the Python logic.

Run with:
    pytest tests/unit/test_layernorm.py -v -s
"""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi.layers.layernorm import LayerNorm


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_inputs(B: int, ni: int, seed: int = 0):
    torch.manual_seed(seed)
    ma = torch.randn(B, ni, device=DEVICE)
    Sa = torch.rand(B, ni, device=DEVICE).abs() * 0.1 + 1e-4
    return ma, Sa


def _make_deltas(B: int, ni: int, seed: int = 1):
    torch.manual_seed(seed)
    delta_mz = torch.randn(B, ni, device=DEVICE)
    delta_Sz = torch.rand(B, ni, device=DEVICE).abs() * 0.1 + 1e-4
    return delta_mz, delta_Sz


# ===========================================================================
#  Forward pass
# ===========================================================================


class TestLayerNormForward:
    def test_output_shape(self):
        B, ni = 8, 16
        layer = LayerNorm(ni, device=DEVICE)
        ma, Sa = _make_inputs(B, ni)
        mz, Sz = layer.forward(ma, Sa)
        assert mz.shape == (B, ni)
        assert Sz.shape == (B, ni)

    def test_zero_input_zero_output_mean(self):
        """With all-zero inputs, normalized output means are zero (bias=0 init)."""
        B, ni = 4, 8
        layer = LayerNorm(ni, device=DEVICE)
        # Override mb to zero for this test
        layer.mb = torch.zeros(ni, device=DEVICE)
        ma = torch.zeros(B, ni, device=DEVICE)
        Sa = torch.ones(B, ni, device=DEVICE) * 0.1
        mz, _ = layer.forward(ma, Sa)
        assert torch.allclose(mz, torch.zeros_like(mz), atol=1e-5)

    def test_output_var_non_negative(self):
        B, ni = 8, 32
        layer = LayerNorm(ni, device=DEVICE)
        ma, Sa = _make_inputs(B, ni)
        _, Sz = layer.forward(ma, Sa)
        assert torch.all(Sz >= 0)

    def test_normalized_mean_approx_zero(self):
        """With mw=1, mb=0: output means should have ~zero mean along features."""
        B, ni = 16, 64
        layer = LayerNorm(ni, device=DEVICE)
        layer.mw = torch.ones(ni, device=DEVICE)
        layer.mb = torch.zeros(ni, device=DEVICE)
        layer.Sw = torch.zeros(ni, device=DEVICE)
        layer.Sb = torch.zeros(ni, device=DEVICE)
        ma, Sa = _make_inputs(B, ni, seed=42)
        mz, _ = layer.forward(ma, Sa)
        # Each sample's output mean should be ~0 (mean of normalized vector)
        sample_means = mz.mean(dim=1)
        assert torch.allclose(sample_means, torch.zeros(B, device=DEVICE), atol=1e-5)

    def test_normalized_std_approx_one(self):
        """With mw=1, mb=0, Sw=0: output std per sample ≈ 1."""
        B, ni = 16, 128
        layer = LayerNorm(ni, device=DEVICE)
        layer.mw = torch.ones(ni, device=DEVICE)
        layer.mb = torch.zeros(ni, device=DEVICE)
        layer.Sw = torch.zeros(ni, device=DEVICE)
        layer.Sb = torch.zeros(ni, device=DEVICE)
        ma, _ = _make_inputs(B, ni, seed=7)
        Sa = torch.zeros(B, ni, device=DEVICE)  # zero input variance for cleaner test
        mz, _ = layer.forward(ma, Sa)
        # Std of the normalized output across features ≈ 1 (Bessel-corrected)
        sample_stds = mz.std(dim=1, correction=1)
        assert torch.allclose(sample_stds, torch.ones(B, device=DEVICE), atol=2e-3)

    def test_deterministic(self):
        B, ni = 4, 16
        layer = LayerNorm(ni, device=DEVICE)
        ma, Sa = _make_inputs(B, ni)
        mz1, Sz1 = layer.forward(ma, Sa)
        mz2, Sz2 = layer.forward(ma, Sa)
        torch.testing.assert_close(mz1, mz2)
        torch.testing.assert_close(Sz1, Sz2)

    def test_gamma_scale(self):
        """Doubling mw doubles the output means."""
        B, ni = 4, 8
        ma, Sa = _make_inputs(B, ni)

        layer1 = LayerNorm(ni, device=DEVICE)
        layer1.mw = torch.ones(ni, device=DEVICE)
        layer1.Sw = torch.zeros(ni, device=DEVICE)
        layer1.mb = torch.zeros(ni, device=DEVICE)
        layer1.Sb = torch.zeros(ni, device=DEVICE)
        mz1, _ = layer1.forward(ma, Sa)

        layer2 = LayerNorm(ni, device=DEVICE)
        layer2.mw = torch.ones(ni, device=DEVICE) * 2.0
        layer2.Sw = torch.zeros(ni, device=DEVICE)
        layer2.mb = torch.zeros(ni, device=DEVICE)
        layer2.Sb = torch.zeros(ni, device=DEVICE)
        mz2, _ = layer2.forward(ma, Sa)

        torch.testing.assert_close(mz2, mz1 * 2.0, atol=1e-5, rtol=0)

    def test_beta_shift(self):
        """Setting mb = c shifts every output by c."""
        B, ni = 4, 8
        c = 3.14
        ma, Sa = _make_inputs(B, ni)

        layer0 = LayerNorm(ni, device=DEVICE)
        layer0.mb = torch.zeros(ni, device=DEVICE)
        mz0, _ = layer0.forward(ma, Sa)

        layer1 = LayerNorm(ni, device=DEVICE)
        layer1.mb = torch.full((ni,), c, device=DEVICE)
        mz1, _ = layer1.forward(ma, Sa)

        torch.testing.assert_close(mz1, mz0 + c, atol=1e-5, rtol=0)

    def test_cache_populated_after_forward(self):
        B, ni = 4, 16
        layer = LayerNorm(ni, device=DEVICE)
        ma, Sa = _make_inputs(B, ni)
        layer.forward(ma, Sa)
        assert layer.ma_in is not None
        assert layer.mu_ra is not None
        assert layer.var_ra is not None
        assert layer.ma_in.shape == (B, ni)
        assert layer.mu_ra.shape == (B, 1)
        assert layer.var_ra.shape == (B, 1)

    def test_bessel_correction(self):
        """var_ra should match Bessel-corrected sample variance of input means plus uncertainty."""
        B, ni = 2, 8
        layer = LayerNorm(ni, device=DEVICE)
        ma = torch.arange(float(ni), device=DEVICE).unsqueeze(0).expand(B, ni)
        Sa = torch.zeros(B, ni, device=DEVICE)
        layer.forward(ma, Sa)

        mu_ra = ma.mean(dim=1, keepdim=True)
        expected_var_ra = ((ma - mu_ra) ** 2).sum(dim=1, keepdim=True) / (ni - 1)
        torch.testing.assert_close(layer.var_ra, expected_var_ra, atol=1e-5, rtol=0)


# ===========================================================================
#  Backward pass
# ===========================================================================


class TestLayerNormBackward:
    def test_backward_output_shape(self):
        B, ni = 8, 16
        layer = LayerNorm(ni, device=DEVICE)
        ma, Sa = _make_inputs(B, ni)
        layer.forward(ma, Sa)
        delta_mz, delta_Sz = _make_deltas(B, ni)
        d_ma, d_Sa = layer.backward(delta_mz, delta_Sz)
        assert d_ma.shape == (B, ni)
        assert d_Sa.shape == (B, ni)

    def test_delta_Sa_sign(self):
        """delta_Sa must be non-positive (variance innovation reduces uncertainty)."""
        B, ni = 8, 16
        layer = LayerNorm(ni, device=DEVICE)
        ma, Sa = _make_inputs(B, ni)
        layer.forward(ma, Sa)
        _, delta_Sz = _make_deltas(B, ni)
        # Use negative delta_Sz to produce negative delta_Sa (normal training signal)
        _, d_Sa = layer.backward(torch.zeros(B, ni, device=DEVICE), -delta_Sz.abs())
        assert torch.all(d_Sa <= 0)

    def test_backward_deterministic(self):
        B, ni = 4, 8
        layer = LayerNorm(ni, device=DEVICE)
        ma, Sa = _make_inputs(B, ni)
        layer.forward(ma, Sa)
        delta_mz, delta_Sz = _make_deltas(B, ni)
        d1 = layer.backward(delta_mz, delta_Sz)
        d2 = layer.backward(delta_mz, delta_Sz)
        torch.testing.assert_close(d1[0], d2[0])
        torch.testing.assert_close(d1[1], d2[1])

    def test_parameter_deltas_populated(self):
        B, ni = 8, 16
        layer = LayerNorm(ni, device=DEVICE)
        ma, Sa = _make_inputs(B, ni)
        layer.forward(ma, Sa)
        delta_mz, delta_Sz = _make_deltas(B, ni)
        layer.backward(delta_mz, delta_Sz)
        assert layer.delta_mw is not None and layer.delta_mw.shape == (ni,)
        assert layer.delta_Sw is not None and layer.delta_Sw.shape == (ni,)
        assert layer.delta_mb is not None and layer.delta_mb.shape == (ni,)
        assert layer.delta_Sb is not None and layer.delta_Sb.shape == (ni,)

    def test_zero_gamma_gives_zero_delta_z(self):
        """If mw = 0, no signal should propagate backward."""
        B, ni = 4, 8
        layer = LayerNorm(ni, device=DEVICE)
        layer.mw = torch.zeros(ni, device=DEVICE)
        ma, Sa = _make_inputs(B, ni)
        layer.forward(ma, Sa)
        delta_mz, delta_Sz = _make_deltas(B, ni)
        d_ma, d_Sa = layer.backward(delta_mz, delta_Sz)
        assert torch.all(d_ma == 0)
        assert torch.all(d_Sa == 0)

    def test_manual_forward_backward_consistency(self):
        """Manually verify one sample against the formula."""
        B, ni = 1, 4
        eps = 1e-5
        mw = torch.tensor([1.0, 2.0, 0.5, -1.0], device=DEVICE)
        ma = torch.tensor([[2.0, -1.0, 0.5, 3.0]], device=DEVICE)
        Sa = torch.tensor([[0.1, 0.2, 0.05, 0.3]], device=DEVICE)

        layer = LayerNorm(ni, eps=eps, device=DEVICE)
        layer.mw = mw.clone()
        layer.Sw = torch.zeros(ni, device=DEVICE)
        layer.mb = torch.zeros(ni, device=DEVICE)
        layer.Sb = torch.zeros(ni, device=DEVICE)
        layer.forward(ma, Sa)

        mu_ra = ma.mean(dim=1, keepdim=True)
        var_s = Sa.sum(dim=1, keepdim=True)
        var_ra = ((ma - mu_ra) ** 2).sum(dim=1, keepdim=True) + var_s
        var_ra = var_ra / (ni - 1)
        inv_std = 1.0 / (var_ra + eps).sqrt()
        expected_inv_std = inv_std.expand(B, ni)

        torch.testing.assert_close(layer.var_ra, var_ra, atol=1e-5, rtol=0)

        # Check Jacobian (diagonal approximation): tmp = inv_std * mw
        delta_mz = torch.ones(B, ni, device=DEVICE)
        delta_Sz = torch.ones(B, ni, device=DEVICE)
        d_ma, d_Sa = layer.backward(delta_mz, delta_Sz)

        expected_tmp = expected_inv_std * mw
        torch.testing.assert_close(d_ma, expected_tmp * delta_mz, atol=1e-5, rtol=0)
        torch.testing.assert_close(d_Sa, expected_tmp**2 * delta_Sz, atol=1e-5, rtol=0)


# ===========================================================================
#  Update pass
# ===========================================================================


class TestLayerNormUpdate:
    def test_update_changes_mw(self):
        B, ni = 8, 16
        layer = LayerNorm(ni, device=DEVICE)
        ma, Sa = _make_inputs(B, ni)
        layer.forward(ma, Sa)
        delta_mz, delta_Sz = _make_deltas(B, ni)
        layer.backward(delta_mz, delta_Sz)

        mw_before = layer.mw.clone()
        layer.update(cap_factor=2.0)
        # mw should have changed (not all delta_mw will be zero)
        assert not torch.all(layer.mw == mw_before)

    def test_update_keeps_Sw_positive(self):
        B, ni = 8, 16
        layer = LayerNorm(ni, device=DEVICE)
        ma, Sa = _make_inputs(B, ni)
        layer.forward(ma, Sa)
        delta_mz, delta_Sz = _make_deltas(B, ni)
        layer.backward(delta_mz, delta_Sz)
        layer.update(cap_factor=2.0)
        assert torch.all(layer.Sw > 0)
        assert torch.all(layer.Sb > 0)

    def test_num_parameters(self):
        ni = 32
        layer = LayerNorm(ni, device=DEVICE)
        # 2 * 2 * ni: (mw, Sw) + (mb, Sb)
        assert layer.num_parameters == 4 * ni
