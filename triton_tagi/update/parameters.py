"""
Parameter update — capped Bayesian update matching cuTAGI.

The update rule is:
    delta_bar  = √S / cap_factor       (adaptive cap per-parameter)
    m_new = m + sign(Δ_μ) · min(|Δ_μ|, delta_bar)
    S_new = S + sign(Δ_S) · min(|Δ_S|, delta_bar)   (if result > 0)
    S_new = 1e-5                                       (only if result ≤ 0)

Cap factor is a heuristic that regularises updates for larger batches:
    batch == 1:    cap_factor = 0.1
    1 < batch < 256:  cap_factor = 2.0
    batch >= 256:  cap_factor = 3.0

This is a general-purpose function — it works on any parameter tensor
(weights, biases, or any future learnable parameters).

Originally a fused Triton kernel; this is a pure-PyTorch port with identical
math, running in-place on CPU (and CUDA/MPS if available).
"""

import torch


# ======================================================================
#  Cap-factor heuristic (matches cuTAGI)
# ======================================================================


def get_cap_factor(batch_size: int) -> float:
    """
    Get the cap factor for regularising parameter updates.

    Based on empirical tuning in cuTAGI — larger batches need stronger
    regularisation to prevent overshooting.

    Parameters
    ----------
    batch_size : int

    Returns
    -------
    cap_factor : float
    """
    if batch_size == 1:
        return 0.1
    elif batch_size < 256:
        return 2.0
    else:
        return 3.0


# ======================================================================
#  Python API
# ======================================================================


def update_parameters(m, S, delta_m, delta_S, cap_factor):
    """
    In-place capped Bayesian parameter update (matches cuTAGI).

    Each update is independently capped at delta_bar = √S / cap_factor,
    preventing large updates when the batch is large.

    Parameters
    ----------
    m          : Tensor  parameter means    (modified in-place)
    S          : Tensor  parameter variances (modified in-place)
    delta_m    : Tensor  mean deltas    (Sw * grad_m)
    delta_S    : Tensor  variance deltas (Sw² * grad_S)
    cap_factor : float   regularisation strength
    """
    # Adaptive cap computed from the *current* variance, before any update.
    delta_bar = torch.sqrt(S.clamp_min(1e-10)) / cap_factor

    # ── Capped mean update ──
    dm_capped = torch.sign(delta_m) * torch.minimum(delta_m.abs(), delta_bar)
    m.add_(dm_capped)

    # ── Capped variance update ──
    # cuTAGI floors S at 1e-5 only when the update would make it non-positive
    # (base_layer.cpp: `if (var_w[i] <= 0.0f) var_w[i] = 1E-5f`).
    dS_capped = torch.sign(delta_S) * torch.minimum(delta_S.abs(), delta_bar)
    S_raw = S + dS_capped
    S.copy_(torch.where(S_raw <= 0.0, torch.full_like(S_raw, 1e-5), S_raw))
