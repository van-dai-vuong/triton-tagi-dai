"""
Attention variance-path kernels (pure-PyTorch port).

The MultiheadAttentionV2 forward and backward each do three structurally
identical variance computations. These were originally fused Triton kernels;
this module provides the equivalent pure-PyTorch batched-matmul implementations
so the layer runs on CPU (and CUDA/MPS if available). The math and public
function signatures are unchanged.

For every kernel, the forward is expressed in TAGI-product form:
    sum over k:    mean = μ_a · μ_b
                   var  = var_a · var_b + var_a · μ_b² + μ_a² · var_b
                        = var_a · (μ_b² + var_b)  +  μ_a² · var_b

``bmm_tagi_var``
    Full TAGI-product variance ``var_ab`` for two batched Gaussian tensors
    (both operands have nontrivial variance). Used for ``QKᵀ`` and
    ``Score @ V`` in the forward pass.

``bmm_shared_right`` / ``bmm_shared_left``
    Both ``mean`` and ``var`` when one operand is deterministic (no variance):
    the deterministic operand is squared for the variance path. Used for all
    four backward reductions (``δV / δscore / δQ / δK``).

``torch.matmul`` broadcasts over all leading (batch) dims, so views produced by
``.transpose(-1, -2)`` are handled directly without any manual stride juggling.
"""

from __future__ import annotations

import torch


# ======================================================================
#  TAGI-product variance, both operands Gaussian
#
#  var_ab[..., m, l] = scale_sq · Σ_k ( var_a · (μ_b² + var_b) + μ_a² · var_b )
# ======================================================================


def bmm_tagi_var(
    mu_a: torch.Tensor,
    var_a: torch.Tensor,
    mu_b: torch.Tensor,
    var_b: torch.Tensor,
    scale_sq: float = 1.0,
) -> torch.Tensor:
    """TAGI-product variance for batched matmul.

    Computes the last-two-dim matmul variance for every batch slice::

        var_ab[..., m, l] = scale_sq · Σ_k ( var_a · (μ_b² + var_b)
                                             + μ_a² · var_b )

    Args:
        mu_a, var_a: shape (..., M, K)
        mu_b, var_b: shape (..., K, L)
        scale_sq:    scalar multiplier applied after reduction
                     (e.g. ``1/head_dim`` for scaled dot-product attention)

    Returns:
        var_ab: shape (..., M, L).
    """
    if mu_a.shape[-1] != mu_b.shape[-2]:
        raise ValueError(
            f"Reduction dims disagree: a has {mu_a.shape[-1]}, b has {mu_b.shape[-2]}"
        )
    out = torch.matmul(var_a, mu_b * mu_b + var_b) + torch.matmul(mu_a * mu_a, var_b)
    return out * scale_sq


# ======================================================================
#  Shared deterministic operand
#
#  mean_out[..., m, l] = Σ_k  a_mean[..., m, k] · b[..., k, l]
#  var_out [..., m, l] = Σ_k  a_var [..., m, k] · b[..., k, l]²
# ======================================================================


def bmm_shared_right(
    a_mean: torch.Tensor,
    a_var: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched matmul with a deterministic right operand.

    Returns both the mean and variance outputs::

        mean[..., m, l] = scale   · Σ_k a_mean[..., m, k] · b[..., k, l]
        var [..., m, l] = scale²  · Σ_k a_var [..., m, k] · b[..., k, l]²

    Args:
        a_mean, a_var: shape (..., M, K)
        b:             shape (..., K, L)
        scale:         scalar applied to the mean (variance gets ``scale²``)

    Returns:
        (mean, var), each of shape (..., M, L).
    """
    if a_mean.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"Reduction dims disagree: a has {a_mean.shape[-1]}, b has {b.shape[-2]}"
        )
    mean = torch.matmul(a_mean, b) * scale
    var = torch.matmul(a_var, b * b) * (scale * scale)
    return mean, var


def bmm_shared_left(
    a: torch.Tensor,
    b_mean: torch.Tensor,
    b_var: torch.Tensor,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched matmul with a deterministic left operand.

    Returns both the mean and variance outputs::

        mean[..., m, l] = scale  · Σ_k a[..., m, k]  · b_mean[..., k, l]
        var [..., m, l] = scale² · Σ_k a[..., m, k]² · b_var [..., k, l]

    Args:
        a:             shape (..., M, K)
        b_mean, b_var: shape (..., K, L)
        scale:         scalar applied to the mean (variance gets ``scale²``)

    Returns:
        (mean, var), each of shape (..., M, L).
    """
    if a.shape[-1] != b_mean.shape[-2]:
        raise ValueError(
            f"Reduction dims disagree: a has {a.shape[-1]}, b has {b_mean.shape[-2]}"
        )
    mean = torch.matmul(a, b_mean) * scale
    var = torch.matmul(a * a, b_var) * (scale * scale)
    return mean, var
