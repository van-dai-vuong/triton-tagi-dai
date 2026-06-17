"""Tests for the quadrature-based Laplace-Remax experiment path."""

from __future__ import annotations

import torch

from triton_tagi.layers.remax import Remax, laplace_remax


def _sobol_reference(mu: torch.Tensor, var: torch.Tensor, n: int = 1 << 18):
    """Deterministic quasi-Monte Carlo reference for the rectified ratio."""
    K = mu.numel()
    engine = torch.quasirandom.SobolEngine(K, scramble=False)
    u = engine.draw(n).to(dtype=torch.float64).clamp_(1e-12, 1.0 - 1e-12)
    z = mu + torch.sqrt(var) * torch.special.ndtri(u)
    m = torch.clamp(z, min=0.0)
    s = m.sum(dim=1, keepdim=True)
    a = torch.where(s > 0.0, m / s, torch.full_like(m, 1.0 / K))

    mu_a = a.mean(dim=0)
    var_a = torch.mean(a * a, dim=0) - mu_a * mu_a
    eaz = torch.einsum("ni,nj->ij", a, z) / n
    jac = (eaz - mu_a.unsqueeze(1) * mu.unsqueeze(0)) / var.unsqueeze(0)
    return mu_a, var_a, jac


def test_laplace_remax_matches_gauss_hermite_reference():
    mu = torch.tensor([0.35, -0.15, 0.05], dtype=torch.float64)
    var = torch.tensor([0.45, 0.30, 0.60], dtype=torch.float64)

    ma, Sa, J = laplace_remax(
        mu,
        var,
        num_quad=80,
        full_jacobian=True,
        cap_jacobian=False,
    )
    ref_ma, ref_Sa, ref_J = _sobol_reference(mu, var)

    torch.testing.assert_close(ma, ref_ma, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(Sa, ref_Sa, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(J, ref_J, atol=4e-3, rtol=4e-3)
    torch.testing.assert_close(ma.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-6, rtol=0)


def test_laplace_remax_all_zero_policy_keeps_uniform_mass():
    mu = torch.full((1, 4), -10.0, dtype=torch.float64)
    var = torch.full_like(mu, 1e-4)

    ma, Sa, J = laplace_remax(mu, var, num_quad=48, full_jacobian=True)

    torch.testing.assert_close(ma, torch.full_like(ma, 0.25), atol=1e-6, rtol=0)
    torch.testing.assert_close(Sa, torch.zeros_like(Sa), atol=1e-6, rtol=0)
    assert J.shape == (1, 4, 4)
    assert torch.isfinite(J).all()


def test_laplace_remax_full_backward_uses_output_input_orientation():
    layer = Remax(approximation="laplace", jacobian="full", num_quad=48)
    mz = torch.tensor([[0.2, -0.1, 0.4]], dtype=torch.float64)
    Sz = torch.tensor([[0.5, 0.4, 0.3]], dtype=torch.float64)
    layer.forward(mz, Sz)

    delta_ma = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float64)
    delta_Sa = torch.tensor([[0.3, 0.1, 0.2]], dtype=torch.float64)
    delta_mz, delta_Sz = layer.backward(delta_ma, delta_Sa)

    expected_mz = torch.einsum("bi,bij->bj", delta_ma, layer.J)
    expected_Sz = torch.einsum("bi,bij->bj", delta_Sa, layer.J * layer.J)
    torch.testing.assert_close(delta_mz, expected_mz)
    torch.testing.assert_close(delta_Sz, expected_Sz)
