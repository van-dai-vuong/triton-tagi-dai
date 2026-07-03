"""
Observation update — compute the innovation signal at the output layer.

Given observations y, predicted mean μ_z and predicted variance S_z,
the TAGI update computes:
    δ_μ = (y − μ_z) / (S_z + σ_v²)
    δ_S = −1      / (S_z + σ_v²)

These deltas are then back-propagated through the network.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024


# ======================================================================
#  Triton kernel
# ======================================================================


@triton.jit
def _output_innovation_kernel(
    y_ptr,
    ym_ptr,
    yS_ptr,
    sv_sq,
    dm_ptr,
    dS_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    y = tl.load(y_ptr + offs, mask=valid)
    ym = tl.load(ym_ptr + offs, mask=valid)
    yS = tl.load(yS_ptr + offs, mask=valid)

    Sy = yS + sv_sq

    tl.store(dm_ptr + offs, (y - ym) / Sy, mask=valid)
    tl.store(dS_ptr + offs, -1.0 / Sy, mask=valid)


@triton.jit
def _output_innovation_kernel_heteros(
    y_ptr,
    ym_ptr,
    yS_ptr,
    dm_ptr,
    dS_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    """AGVI heteroscedastic output update — 1:1 port of cuTAGI's CUDA
    ``update_delta_z_cuda_heteros`` (``src/output_updater_cuda.cu``).

    The output layer is interleaved width ``2K``: column ``2i`` is the mean
    prediction ``Z_i`` (identity activation, Jacobian = 1), column ``2i+1`` is
    the post-:class:`~triton_tagi.layers.EvenExp` aleatoric variance ``V̄²_i``
    (log-normal moments). Terminology follows cuTAGI: ``V ~ N(0, μ_V2)`` is the
    error, ``V2 = V²``, ``V2_bar`` its expectation, ``V2_bar_tilde = exp(·)`` the
    positive-domain activation output.

    Because EvenExp's Jacobian equals its mean (``jcb = μ_a``), the smoother gain
    that maps the V̄²-tilde posterior back to the pre-activation latent is
    ``jv = μ_{V̄²} / Σ_{V̄²}`` — folded in HERE, exactly as cuTAGI does. The paired
    ``EvenExp.backward`` is therefore an identity passthrough.

    Note: cuTAGI's CUDA kernel updates the mean with the *epistemic* variance
    (``var_a_col``) and the variance with the *total* variance (``var_sum``) —
    the "overfit_mu" behavior. The CPU ``compute_delta_z_heteros`` instead uses
    ``var_sum`` for both. We follow the CUDA path, which is what the cuTAGI
    regression-heteros example (``cuda=True``) actually runs.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements

    y = tl.load(y_ptr + offs, mask=valid)

    # Interleaved 2K output: even = Z (mean), odd = V̄²_tilde (post-EvenExp).
    obs_col = offs * 2

    mu_a_col = tl.load(ym_ptr + obs_col, mask=valid)
    var_a_col = tl.load(yS_ptr + obs_col, mask=valid)

    # V2_bar_tilde moments. For EvenExp, jcb = μ_a, so cov_V2_bar_tilde = μ.
    mu_v2_bar_tilde = tl.load(ym_ptr + obs_col + 1, mask=valid)
    var_v2_bar_tilde = tl.load(yS_ptr + obs_col + 1, mask=valid)
    cov_v2_bar_tilde = mu_v2_bar_tilde

    # Prior predictive for V2 (Gaussian-moment chain).
    mu_v2 = mu_v2_bar_tilde
    var_v2 = 3.0 * var_v2_bar_tilde + 2.0 * mu_v2_bar_tilde * mu_v2_bar_tilde
    cov_y_v = mu_v2

    # Total output variance: epistemic (var_a_col) + learned aleatoric (mu_v2).
    var_sum = var_a_col + mu_v2

    # ── Z (even) update: jcb_col = 1 (identity). Mean uses epistemic variance,
    #    variance uses total variance — cuTAGI's overfit_mu behavior. ──
    obs_diff = y - mu_a_col
    tmp_mu = 1.0 / var_a_col
    tmp_var = 1.0 / var_sum
    z_ok = (var_a_col > 0.0) & (var_sum > 0.0)
    delta_mu_col = tl.where(z_ok, tmp_mu * obs_diff, 0.0)
    delta_var_col = tl.where(z_ok, -tmp_var, 0.0)

    # ── V̄² (odd) AGVI update ──
    mu_v_post = cov_y_v / var_sum * obs_diff
    var_v_post = mu_v2 - cov_y_v / var_sum * cov_y_v

    mu_v2_post = mu_v_post * mu_v_post + var_v_post
    var_v2_post = 2.0 * var_v_post * var_v_post + 4.0 * var_v_post * mu_v_post * mu_v_post

    tmp_ratio = var_v2_bar_tilde / var_v2
    mu_v2_bar_tilde_post = mu_v2_bar_tilde + tmp_ratio * (mu_v2_post - mu_v2)
    var_v2_bar_tilde_post = var_v2_bar_tilde + tmp_ratio * tmp_ratio * (var_v2_post - var_v2)

    # Fold the exp Jacobian (smoother gain) into the pre-activation delta.
    jv = cov_v2_bar_tilde / var_v2_bar_tilde
    delta_mu_v2 = jv * (mu_v2_bar_tilde_post - mu_v2_bar_tilde)
    delta_var_v2 = jv * jv * (var_v2_bar_tilde_post - var_v2_bar_tilde)

    tl.store(dm_ptr + obs_col, delta_mu_col, mask=valid)
    tl.store(dS_ptr + obs_col, delta_var_col, mask=valid)

    tl.store(dm_ptr + obs_col + 1, delta_mu_v2, mask=valid)
    tl.store(dS_ptr + obs_col + 1, delta_var_v2, mask=valid)


# ======================================================================
#  Python API
# ======================================================================


def compute_innovation(y, y_pred_mu, y_pred_var, sigma_v):
    """
    Compute the output innovation (update signal) for TAGI.

    Parameters
    ----------
    y          : Tensor (B, D)  observed targets
    y_pred_mu  : Tensor (B, D)  predicted output mean
    y_pred_var : Tensor (B, D)  predicted output variance
    sigma_v    : float          observation noise std-dev

    Returns
    -------
    delta_mu  : Tensor (B, D)  mean innovation
    delta_var : Tensor (B, D)  variance innovation
    """
    n = y.numel()

    if y_pred_mu.shape[-1] == 2 * y.shape[-1]:
        delta_mu = torch.empty_like(y_pred_mu)
        delta_var = torch.empty_like(y_pred_var)
        _output_innovation_kernel_heteros[(triton.cdiv(n, BLOCK),)](
            y,
            y_pred_mu,
            y_pred_var,
            delta_mu,
            delta_var,
            n,
            BLOCK=BLOCK,
        )
    else:
        delta_mu = torch.empty_like(y)
        delta_var = torch.empty_like(y)
        _output_innovation_kernel[(triton.cdiv(n, BLOCK),)](
            y,
            y_pred_mu,
            y_pred_var,
            sigma_v**2,
            delta_mu,
            delta_var,
            n,
            BLOCK=BLOCK,
        )

    return delta_mu, delta_var


# ======================================================================
#  Sparse (hierarchical softmax) output innovation
# ======================================================================


def compute_innovation_with_indices(
    ma: "torch.Tensor",
    Sa: "torch.Tensor",
    y_obs: "torch.Tensor",
    var_obs: "torch.Tensor",
    selected_idx: "torch.Tensor",
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Sparse output innovation for hierarchical softmax classification.

    For each sample b and encoded bit c, updates only the selected tree node::

        node = selected_idx[b, c] - 1          (0-indexed)
        denom = Sa[b, node] + var_obs[b, c]
        delta_mu[b, node] = (y_obs[b, c] - ma[b, node]) / denom
        delta_Sa[b, node] = -1 / denom

    All other positions in delta_mu and delta_Sa are zero.

    Replicates cuTAGI's ``compute_selected_delta_z_output()`` from
    ``src/base_output_updater.cpp``.

    Args:
        ma:           Output means, shape (B, n_total_nodes).
        Sa:           Output variances, shape (B, n_total_nodes).
        y_obs:        Encoded ±1 observations, shape (B, n_obs).
        var_obs:      Observation variance, shape (B, n_obs).
        selected_idx: 1-indexed node positions, shape (B, n_obs).

    Returns:
        delta_mu: Mean innovations, shape (B, n_total_nodes), sparse.
        delta_Sa: Variance innovations, shape (B, n_total_nodes), sparse.
    """
    delta_mu = torch.zeros_like(ma)
    delta_Sa = torch.zeros_like(Sa)

    # Convert 1-indexed to 0-indexed: (B, n_obs)
    node_idx = selected_idx.long() - 1

    # Gather predicted mean and variance at selected nodes
    ma_sel = torch.gather(ma, 1, node_idx)  # (B, n_obs)
    Sa_sel = torch.gather(Sa, 1, node_idx)  # (B, n_obs)

    # Innovation formula (same as dense case, evaluated at selected nodes only)
    denom = Sa_sel + var_obs
    dm = (y_obs - ma_sel) / denom
    dS = -1.0 / denom

    # Scatter innovations back into the full output buffers.
    # Each class uses distinct tree nodes at every depth level, so no
    # within-sample index collision occurs for valid HRC trees.
    delta_mu.scatter_add_(1, node_idx, dm)
    delta_Sa.scatter_add_(1, node_idx, dS)

    return delta_mu, delta_Sa
