"""Unit tests for triton_tagi.layers.maxpool2d.MaxPool2D.

No cuTAGI dependency — tests the hard-argmax moment propagation.

Run with:
    pytest tests/unit/test_maxpool2d.py -v
"""

from __future__ import annotations

import pytest
import torch

from triton_tagi.layers.maxpool2d import MaxPool2D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_inputs(N, C, H, W, seed=0):
    torch.manual_seed(seed)
    ma = torch.randn(N, C, H, W, device=DEVICE)
    Sa = torch.rand(N, C, H, W, device=DEVICE).abs() * 0.1 + 1e-4
    return ma, Sa


# ===========================================================================
#  Forward pass
# ===========================================================================


class TestMaxPool2DForward:
    def test_output_shape_no_overlap(self):
        """stride=kernel_size: H_out = H // k."""
        N, C, H, W, k = 2, 4, 8, 8, 2
        layer = MaxPool2D(k, stride=k)
        ma, Sa = _make_inputs(N, C, H, W)
        mz, Sz = layer.forward(ma, Sa)
        assert mz.shape == (N, C, H // k, W // k)
        assert Sz.shape == (N, C, H // k, W // k)

    def test_output_shape_stride1(self):
        """stride=1, padding=k//2: same spatial size."""
        N, C, H, W, k = 2, 4, 8, 8, 3
        layer = MaxPool2D(k, stride=1, padding=k // 2)
        ma, Sa = _make_inputs(N, C, H, W)
        mz, Sz = layer.forward(ma, Sa)
        assert mz.shape == (N, C, H, W)
        assert Sz.shape == (N, C, H, W)

    def test_forward_mean_is_max_of_window(self):
        """mu_z[n,c,h,w] must be the maximum mean in the pooling window."""
        N, C, H, W, k = 1, 1, 4, 4, 2
        ma, Sa = _make_inputs(N, C, H, W, seed=7)
        layer = MaxPool2D(k, stride=k)
        mz, _ = layer.forward(ma, Sa)

        import torch.nn.functional as F

        ref, _ = F.max_pool2d_with_indices(ma, k, stride=k)
        torch.testing.assert_close(mz, ref)

    def test_forward_variance_at_argmax(self):
        """var_z must equal the input variance at the argmax position."""
        N, C, H, W, k = 2, 3, 6, 6, 3
        ma, Sa = _make_inputs(N, C, H, W, seed=42)
        layer = MaxPool2D(k, stride=k)
        _, Sz = layer.forward(ma, Sa)

        # Reference: for each output position, find argmax of mu_a, gather Sa
        import torch.nn.functional as F

        H_out, W_out = H // k, W // k
        _, idx = F.max_pool2d_with_indices(ma, k, stride=k)
        Sa_flat = Sa.view(N, C, H * W)
        idx_flat = idx.view(N, C, H_out * W_out)
        Sz_ref = Sa_flat.gather(2, idx_flat).view(N, C, H_out, W_out)
        torch.testing.assert_close(Sz, Sz_ref)

    def test_forward_var_non_negative(self):
        N, C, H, W, k = 4, 8, 8, 8, 2
        ma, Sa = _make_inputs(N, C, H, W)
        layer = MaxPool2D(k, stride=k)
        _, Sz = layer.forward(ma, Sa)
        assert torch.all(Sz >= 0)

    def test_forward_deterministic(self):
        N, C, H, W, k = 2, 4, 8, 8, 2
        ma, Sa = _make_inputs(N, C, H, W)
        layer = MaxPool2D(k, stride=k)
        mz1, Sz1 = layer.forward(ma, Sa)
        mz2, Sz2 = layer.forward(ma, Sa)
        torch.testing.assert_close(mz1, mz2)
        torch.testing.assert_close(Sz1, Sz2)

    def test_cache_populated_after_forward(self):
        N, C, H, W, k = 2, 4, 8, 8, 2
        ma, Sa = _make_inputs(N, C, H, W)
        layer = MaxPool2D(k, stride=k)
        layer.forward(ma, Sa)
        assert layer.pool_idx is not None
        assert layer.input_shape == (N, C, H, W)
        assert layer.pool_idx.shape == (N, C, H // k, W // k)

    def test_forward_single_element_pool(self):
        """1x1 pooling is identity."""
        N, C, H, W = 2, 3, 4, 4
        ma, Sa = _make_inputs(N, C, H, W)
        layer = MaxPool2D(1, stride=1)
        mz, Sz = layer.forward(ma, Sa)
        torch.testing.assert_close(mz, ma)
        torch.testing.assert_close(Sz, Sa)


# ===========================================================================
#  Backward pass
# ===========================================================================


class TestMaxPool2DBackward:
    def test_backward_output_shape(self):
        N, C, H, W, k = 2, 4, 8, 8, 2
        ma, Sa = _make_inputs(N, C, H, W)
        layer = MaxPool2D(k, stride=k)
        layer.forward(ma, Sa)
        torch.manual_seed(1)
        dm = torch.randn(N, C, H // k, W // k, device=DEVICE)
        ds = torch.rand(N, C, H // k, W // k, device=DEVICE).abs() * 0.1
        d_ma, d_Sa = layer.backward(dm, ds)
        assert d_ma.shape == (N, C, H, W)
        assert d_Sa.shape == (N, C, H, W)

    def test_backward_routes_to_argmax_only(self):
        """Delta must be non-zero only at the argmax position in each window."""
        N, C, H, W, k = 1, 1, 4, 4, 2
        ma, Sa = _make_inputs(N, C, H, W, seed=0)
        layer = MaxPool2D(k, stride=k)
        layer.forward(ma, Sa)

        H_out, W_out = H // k, W // k
        dm = torch.ones(N, C, H_out, W_out, device=DEVICE)
        ds = torch.ones(N, C, H_out, W_out, device=DEVICE)
        d_ma, _ = layer.backward(dm, ds)

        # Number of non-zero entries must equal H_out * W_out (one per window)
        n_nonzero = (d_ma != 0).sum().item()
        assert n_nonzero == H_out * W_out

    def test_backward_zero_delta_propagates_zeros(self):
        N, C, H, W, k = 2, 4, 8, 8, 2
        ma, Sa = _make_inputs(N, C, H, W)
        layer = MaxPool2D(k, stride=k)
        layer.forward(ma, Sa)
        dm = torch.zeros(N, C, H // k, W // k, device=DEVICE)
        ds = torch.zeros(N, C, H // k, W // k, device=DEVICE)
        d_ma, d_Sa = layer.backward(dm, ds)
        assert torch.all(d_ma == 0)
        assert torch.all(d_Sa == 0)

    def test_backward_delta_sum_preserved(self):
        """Total delta_ma sum == total dm sum (jcb=1 at each position)."""
        N, C, H, W, k = 2, 4, 8, 8, 2
        torch.manual_seed(5)
        ma, Sa = _make_inputs(N, C, H, W)
        layer = MaxPool2D(k, stride=k)
        layer.forward(ma, Sa)
        dm = torch.randn(N, C, H // k, W // k, device=DEVICE)
        ds = torch.rand(N, C, H // k, W // k, device=DEVICE).abs()
        d_ma, d_Sa = layer.backward(dm, ds)
        # Sum must be preserved (scatter_add)
        torch.testing.assert_close(d_ma.sum(), dm.sum(), atol=1e-4, rtol=0)
        torch.testing.assert_close(d_Sa.sum(), ds.sum(), atol=1e-4, rtol=0)

    def test_backward_manual_check(self):
        """Manually verify: for 1 sample, 1 channel, 2x2 input with k=2."""
        # Input: [[3, 1], [2, 4]]
        # Max mean = 4 at position (1,1); Sa at that position = 0.8
        # After forward: mu_z = 4, var_z = 0.8
        # Backward with dm=1: delta_ma should be 1 at (1,1), 0 elsewhere
        ma = torch.tensor([[[[3.0, 1.0], [2.0, 4.0]]]], device=DEVICE)
        Sa = torch.tensor([[[[0.1, 0.2], [0.3, 0.8]]]], device=DEVICE)
        layer = MaxPool2D(2, stride=2)
        mz, Sz = layer.forward(ma, Sa)

        assert mz.item() == pytest.approx(4.0, abs=1e-5)
        assert Sz.item() == pytest.approx(0.8, abs=1e-5)

        dm = torch.tensor([[[[1.0]]]], device=DEVICE)
        ds = torch.tensor([[[[1.0]]]], device=DEVICE)
        d_ma, d_Sa = layer.backward(dm, ds)

        expected_d_ma = torch.tensor([[[[0.0, 0.0], [0.0, 1.0]]]], device=DEVICE)
        torch.testing.assert_close(d_ma, expected_d_ma)

    def test_backward_deterministic(self):
        N, C, H, W, k = 2, 4, 8, 8, 2
        ma, Sa = _make_inputs(N, C, H, W)
        layer = MaxPool2D(k, stride=k)
        layer.forward(ma, Sa)
        torch.manual_seed(2)
        dm = torch.randn(N, C, H // k, W // k, device=DEVICE)
        ds = torch.rand(N, C, H // k, W // k, device=DEVICE).abs()
        d1 = layer.backward(dm, ds)
        d2 = layer.backward(dm, ds)
        torch.testing.assert_close(d1[0], d2[0])
        torch.testing.assert_close(d1[1], d2[1])
