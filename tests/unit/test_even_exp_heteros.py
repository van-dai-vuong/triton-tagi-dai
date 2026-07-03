"""cuTAGI parity tests for the EvenExp variance head + AGVI heteros update.

Ground truth is cuTAGI's CUDA path (what ``examples/regression_heteros.py`` runs
with ``cuda=True``):
  - forward moments  → ``exp_mean_var``               (src/activation.cpp)
  - output update    → ``update_delta_z_cuda_heteros`` (src/output_updater_cuda.cu)

Both are mirrored here in float64 numpy and compared against the triton
implementations within fp32 tolerance (RESEARCH_PRINCIPLES §5: parity is the
floor). The kernel parity test isolates the updater by feeding *the same*
post-activation moments to both sides.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from triton_tagi import EvenExp
from triton_tagi.update.observation import compute_innovation

# fp32 production parity tolerance, per research-hub harness conventions
# (sqrt(eps_fp32) ≈ 3.45e-4); division-amplified, so use a slightly looser rtol.
RTOL = 5e-4
ATOL = 1e-5


# ---------------------------------------------------------------------------
#  cuTAGI numpy reference (float64)
# ---------------------------------------------------------------------------


def ref_exp_mean_var(mz, Sz, scale=1.0, shift=0.0):
    """Mirror of cuTAGI ``exp_mean_var`` (Exp activation moments)."""
    new_mu = mz * scale + shift
    new_var = Sz * scale * scale
    mu_a = np.maximum(np.exp(new_mu + 0.5 * new_var), 1e-6)
    var_a = np.maximum(np.exp(2.0 * new_mu + new_var) * (np.exp(new_var) - 1.0), 1e-6)
    jcb = mu_a * scale
    return mu_a, var_a, jcb


def ref_update_heteros(ym, yS, y):
    """Mirror of cuTAGI ``update_delta_z_cuda_heteros``.

    ``ym``/``yS`` are the *post-activation* interleaved 2K moments
    ``[Z_0, V̄²_0, Z_1, V̄²_1, ...]``; ``y`` is ``(B, K)``. For the exp head,
    ``jcb`` of the variance slot equals its mean, so ``cov_V2_bar_tilde = μ``.
    The Z-slot mean uses the epistemic variance, the variance uses the total.
    """
    B, twoK = ym.shape
    K = twoK // 2
    dm = np.zeros_like(ym)
    dS = np.zeros_like(yS)
    for b in range(B):
        for k in range(K):
            ze, zo = 2 * k, 2 * k + 1
            mu_a_col, var_a_col, jcb_col = ym[b, ze], yS[b, ze], 1.0
            mu_v2bt, var_v2bt = ym[b, zo], yS[b, zo]
            cov_v2bt = mu_v2bt  # jcb = μ_a for exp

            mu_v2 = mu_v2bt
            var_v2 = 3.0 * var_v2bt + 2.0 * mu_v2bt * mu_v2bt
            cov_y_v = mu_v2
            var_sum = var_a_col + mu_v2

            obs_diff = y[b, k] - mu_a_col
            dm[b, ze] = (jcb_col / var_a_col) * obs_diff
            dS[b, ze] = -(jcb_col / var_sum) * jcb_col

            mu_v_post = cov_y_v / var_sum * obs_diff
            var_v_post = mu_v2 - cov_y_v / var_sum * cov_y_v
            mu_v2_post = mu_v_post * mu_v_post + var_v_post
            var_v2_post = 2.0 * var_v_post * var_v_post + 4.0 * var_v_post * mu_v_post * mu_v_post

            tmp_ratio = var_v2bt / var_v2
            mu_v2bt_post = mu_v2bt + tmp_ratio * (mu_v2_post - mu_v2)
            var_v2bt_post = var_v2bt + tmp_ratio * tmp_ratio * (var_v2_post - var_v2)

            jv = cov_v2bt / var_v2bt
            dm[b, zo] = jv * (mu_v2bt_post - mu_v2bt)
            dS[b, zo] = jv * jv * (var_v2bt_post - var_v2bt)
    return dm, dS


# ---------------------------------------------------------------------------
#  Forward moments (CPU, pure torch — no GPU needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale,shift", [(1.0, 0.0), (0.5, 0.2)])
def test_even_exp_forward_matches_exp_mean_var(scale, shift):
    torch.manual_seed(0)
    B, K = 7, 4
    mz = torch.randn(B, 2 * K, dtype=torch.float32)
    Sz = torch.rand(B, 2 * K, dtype=torch.float32) * 2.0 + 1e-3

    ma, Sa = EvenExp(half_width=K, scale=scale, shift=shift).forward(mz, Sz)

    mz_np, Sz_np = mz.double().numpy(), Sz.double().numpy()
    odd = slice(1, None, 2)
    mu_ref, var_ref, _ = ref_exp_mean_var(mz_np[:, odd], Sz_np[:, odd], scale, shift)

    # Odd slots = exp moments; even slots = identity passthrough.
    np.testing.assert_allclose(ma.numpy()[:, odd], mu_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(Sa.numpy()[:, odd], var_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(ma.numpy()[:, ::2], mz_np[:, ::2], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(Sa.numpy()[:, ::2], Sz_np[:, ::2], rtol=RTOL, atol=ATOL)


def test_even_exp_backward_is_identity():
    """cuTAGI folds the exp Jacobian in the updater; the activation backward is
    a passthrough, so deltas must be returned unchanged."""
    K = 3
    dma = torch.randn(5, 2 * K)
    dSa = torch.randn(5, 2 * K)
    out_m, out_S = EvenExp(half_width=K).backward(dma, dSa)
    assert torch.equal(out_m, dma)
    assert torch.equal(out_S, dSa)


# ---------------------------------------------------------------------------
#  Heteroscedastic update kernel parity (needs CUDA)
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@pytest.mark.parametrize("K", [1, 4])
def test_heteros_update_matches_cutagi(K):
    torch.manual_seed(1)
    dev = "cuda"
    B = 16

    # Random pre-activation moments → post-activation via EvenExp (the real head).
    mz = torch.randn(B, 2 * K, device=dev, dtype=torch.float32)
    Sz = (torch.rand(B, 2 * K, device=dev, dtype=torch.float32) * 1.5 + 1e-2)
    ma, Sa = EvenExp(half_width=K).forward(mz, Sz)
    y = torch.randn(B, K, device=dev, dtype=torch.float32)

    # triton (fp32 kernel); sigma_v is ignored by the heteros branch.
    dm, dS = compute_innovation(y, ma, Sa, sigma_v=0.0)

    # reference (fp64) on the SAME post-activation moments — isolates the kernel.
    dm_ref, dS_ref = ref_update_heteros(
        ma.double().cpu().numpy(), Sa.double().cpu().numpy(), y.double().cpu().numpy()
    )

    np.testing.assert_allclose(dm.cpu().numpy(), dm_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(dS.cpu().numpy(), dS_ref, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
