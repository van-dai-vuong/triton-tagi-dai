"""Validation tests: triton-tagi ReLU against fp64 references.

ReLU is also exercised end-to-end in test_mlp.py (as part of a 2-layer MLP vs
cuTAGI). This file validates the layer in isolation so bugs are caught at the
right abstraction level. The fp64 analytical formula is the reference.

Forward reference (exact moments of max(0, z), z ~ N(μ, σ²)):
    α     = μ / σ
    φ(α)  = N(0,1) PDF at α
    Φ(α)  = N(0,1) CDF at α
    μ_a   = σ φ(α) + μ Φ(α)
    S_a   = −μ_a² + 2 μ_a μ − μ σ φ(α) + (S − μ²) Φ(α)
    J     = Φ(α)   (Cauchy–Schwarz clamped in triton)

Backward:
    δ_μ_in = δ_μ_out · J
    δ_S_in = δ_S_out · J²

Run with:
    pytest tests/validation/test_activations.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from triton_tagi.layers.relu import ReLU as TReLU

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FWD_ATOL = 1e-5
BWD_ATOL = 1e-5

pytestmark = pytest.mark.cuda


def _relu_forward_ref(mz: torch.Tensor, Sz: torch.Tensor):
    """Exact moments of max(0, z) in fp64, with Cauchy–Schwarz J clamp."""
    mz64, Sz64 = mz.double(), Sz.double()
    sigma = torch.sqrt(torch.clamp(Sz64, min=1e-12))
    alpha = mz64 / sigma
    phi = torch.exp(-0.5 * alpha ** 2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(alpha / math.sqrt(2.0)))

    mu_a = torch.clamp(sigma * phi + mz64 * Phi, min=1e-7)
    var_a = torch.clamp(
        -mu_a ** 2 + 2.0 * mu_a * mz64 - mz64 * sigma * phi + (Sz64 - mz64 ** 2) * Phi,
        min=1e-7,
    )
    J = Phi
    cs_bound = torch.sqrt(var_a / torch.clamp(Sz64, min=1e-12))
    J = torch.minimum(J, cs_bound)
    return mu_a.float(), var_a.float(), J.float()


def test_relu_forward_mean():
    """ReLU output mean matches fp64 reference."""
    torch.manual_seed(0)
    mz = torch.randn(16, 64, device=DEVICE)
    Sz = torch.rand(16, 64, device=DEVICE).abs() * 0.5 + 1e-4

    layer = TReLU()
    ma, _ = layer.forward(mz, Sz)

    ref_ma, _, _ = _relu_forward_ref(mz.cpu(), Sz.cpu())
    torch.testing.assert_close(ma.cpu(), ref_ma, atol=FWD_ATOL, rtol=0)


def test_relu_forward_variance():
    """ReLU output variance matches fp64 reference."""
    torch.manual_seed(1)
    mz = torch.randn(16, 64, device=DEVICE)
    Sz = torch.rand(16, 64, device=DEVICE).abs() * 0.5 + 1e-4

    layer = TReLU()
    _, Sa = layer.forward(mz, Sz)

    _, ref_Sa, _ = _relu_forward_ref(mz.cpu(), Sz.cpu())
    torch.testing.assert_close(Sa.cpu(), ref_Sa, atol=FWD_ATOL, rtol=0)


def test_relu_backward_delta_mz():
    """ReLU backward δ_μ = δ_μ_out · J matches fp64 reference."""
    torch.manual_seed(0)
    mz = torch.randn(16, 64, device=DEVICE)
    Sz = torch.rand(16, 64, device=DEVICE).abs() * 0.5 + 1e-4
    delta_mz = torch.randn(16, 64, device=DEVICE)
    delta_Sz = torch.rand(16, 64, device=DEVICE).abs() * 0.01

    layer = TReLU()
    layer.forward(mz, Sz)
    d_mz, _ = layer.backward(delta_mz, delta_Sz)

    _, _, J_ref = _relu_forward_ref(mz.cpu(), Sz.cpu())
    ref = delta_mz.cpu() * J_ref
    torch.testing.assert_close(d_mz.cpu(), ref, atol=BWD_ATOL, rtol=0)


def test_relu_backward_delta_Sz():
    """ReLU backward δ_S = δ_S_out · J² matches fp64 reference."""
    torch.manual_seed(1)
    mz = torch.randn(16, 64, device=DEVICE)
    Sz = torch.rand(16, 64, device=DEVICE).abs() * 0.5 + 1e-4
    delta_mz = torch.randn(16, 64, device=DEVICE)
    delta_Sz = torch.rand(16, 64, device=DEVICE).abs() * 0.01

    layer = TReLU()
    layer.forward(mz, Sz)
    _, d_Sz = layer.backward(delta_mz, delta_Sz)

    _, _, J_ref = _relu_forward_ref(mz.cpu(), Sz.cpu())
    ref = delta_Sz.cpu() * J_ref ** 2
    torch.testing.assert_close(d_Sz.cpu(), ref, atol=BWD_ATOL, rtol=0)
