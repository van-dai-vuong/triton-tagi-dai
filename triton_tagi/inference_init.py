"""Inference-Based Initialization (IBI) for TAGI networks.

Pre-training calibration pass. For each Linear layer with width A, the algorithm
drives the empirical batch-aggregate pre-activation moments (mu_Zi, S_Zi) toward
targets derived from layer-sum statistics S = sum_i Zi and S2 = sum_i Zi^2.

Targets (scalar, width-A layer, user-set sigma_m, sigma_z):
    mu_S_tilde   = 0
    var_S_tilde  = A * sigma_z^2
    mu_S2_tilde  = A * (sigma_m^2 + sigma_z^2)
    var_S2_tilde = A * (2 sigma_z^4 + 4 sigma_m^2 sigma_z^2)

Per-layer step on a single batch:
    1. Forward the already-calibrated prefix to obtain (ma, Sa) at this layer.
    2. Forward the Linear to obtain per-sample (mz, Sz) of shape (B, A).
    3. Aggregate to scalars per unit (PLAN.md D1: batch-mean first).
    4. S projection  (closed-form Kalman on S).
    5. S2 RTS update (linearized Kalman on S2, applied after S).
    6. Decoupled inverse: rescale mw, Sw, Sb by gamma_i (gamma_i^2) and shift mb
       by delta_mu_Zi so that the re-forward output moments land on the targets.

Convergence. S hits its target exactly every batch. S2 approaches asymptotically
because the S2 observation is linearized around current (mu_Zi, S_Zi).

Failure modes. If the batch-aggregate var_S or var_S2 falls below eps, the
corresponding projection is skipped for that batch. If S_Zi < eps for a unit,
gamma_i is undefined and that unit is left untouched.

Reference: experiments/inference_init/PLAN.md.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from .layers.linear import Linear
from .network import Sequential


def _layer_targets(
    A: int, sigma_m: float, sigma_z: float
) -> tuple[float, float, float, float]:
    """Return (mu_S_tilde, var_S_tilde, mu_S2_tilde, var_S2_tilde) for width A."""
    var_S = A * sigma_z**2
    mu_S2 = A * (sigma_m**2 + sigma_z**2)
    var_S2 = A * (2.0 * sigma_z**4 + 4.0 * sigma_m**2 * sigma_z**2)
    return 0.0, var_S, mu_S2, var_S2


def _s_projection(
    mu_Z: Tensor,
    S_Z: Tensor,
    mu_S_tilde: float,
    var_S_tilde: float,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Closed-form Kalman update on the scalar observation S = sum_i Zi.

    Skips (returns inputs unchanged) when var_S < eps.
    """
    mu_S = mu_Z.sum()
    var_S = S_Z.sum()
    if float(var_S) < eps:
        return mu_Z, S_Z
    d_mu = (mu_S_tilde - mu_S) / var_S
    d_var = (var_S_tilde - var_S) / (var_S * var_S)
    return mu_Z + S_Z * d_mu, S_Z * (1.0 + var_S * d_var)


def _s2_projection(
    mu_Z: Tensor,
    S_Z: Tensor,
    mu_S2_tilde: float,
    var_S2_tilde: float,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Linearized Kalman update on S2 = sum_i Zi^2, applied after S projection.

    The per-unit Jacobian J_i = 2 mu_Zi S_Zi uses the post-S moments. The post
    variance is floored at 0 because the linearization can over-reduce.
    """
    mu_Z2 = mu_Z * mu_Z + S_Z
    S_Z2 = 2.0 * S_Z * S_Z + 4.0 * S_Z * mu_Z * mu_Z
    mu_S2 = mu_Z2.sum()
    var_S2 = S_Z2.sum()
    if float(var_S2) < eps:
        return mu_Z, S_Z
    J = 2.0 * mu_Z * S_Z
    d_mu = (mu_S2_tilde - mu_S2) / var_S2
    d_var = (var_S2_tilde - var_S2) / (var_S2 * var_S2)
    return mu_Z + J * d_mu, torch.clamp(S_Z + (J * J) * d_var, min=0.0)


def _decoupled_inverse(
    layer: Linear,
    mu_Z: Tensor,
    S_Z: Tensor,
    mu_Z_target: Tensor,
    S_Z_target: Tensor,
    eps: float,
) -> None:
    """Rescale mw, Sw, Sb and shift mb so the post-update forward matches targets.

    gamma_i = sqrt(S_Zi_target / S_Zi).  Applied per output unit:
        mw[:, i]  *= gamma_i
        Sw[:, i]  *= gamma_i^2
        Sb[0, i]  *= gamma_i^2
        mb[0, i]  += mu_Zi_target - (gamma_i (mu_Zi - mb_old) + mb_old)

    Units with S_Zi <= eps are left untouched (gamma_i = 1, delta_mu = 0).
    Modifies ``layer`` in place.
    """
    safe = S_Z > eps
    ratio = torch.where(safe, S_Z_target / S_Z.clamp(min=eps), torch.ones_like(S_Z))
    gamma = torch.sqrt(torch.clamp(ratio, min=0.0))
    gamma = torch.where(safe, gamma, torch.ones_like(gamma))
    mu_target_eff = torch.where(safe, mu_Z_target, mu_Z)

    mb_old = layer.mb.view(-1)
    tilde_mu = gamma * (mu_Z - mb_old) + mb_old
    delta_mu = mu_target_eff - tilde_mu

    g = gamma.unsqueeze(0)
    g2 = (gamma * gamma).unsqueeze(0)
    layer.mw.mul_(g)
    layer.Sw.mul_(g2)
    if layer.has_bias:
        layer.Sb.mul_(g2)
        layer.mb.add_(delta_mu.view(1, -1))


@torch.no_grad()
def inference_init(
    net: Sequential,
    loader: Iterable,
    sigma_m: float,
    sigma_z: float,
    *,
    eps: float = 1e-8,
) -> None:
    """Run one epoch of IBI calibration over every Linear layer in ``net``.

    Non-Linear layers pass through unchanged. Learnable non-Linear layers
    (Conv2D, BatchNorm2D, LayerNorm, ...) are out of scope for V1 and also
    pass through without parameter modification. State (mw, Sw, mb, Sb) on
    Linear layers is modified in place.

    Args:
        net:     Sequential network. Only Linear layers are calibrated in V1.
        loader:  Iterable over one epoch. Each item is either an input Tensor
                 or a tuple whose first element is the input Tensor. Targets
                 are ignored. Inputs are moved to ``net.device`` automatically.
        sigma_m: Global prior mean-scale hyperparameter.
        sigma_z: Global prior activation-scale hyperparameter.
        eps:     Numerical floor for variances (default 1e-8).
    """
    net.eval()
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(net.device)
        ma = x
        Sa = torch.zeros_like(x)

        for layer in net.layers:
            if isinstance(layer, Linear):
                mz, Sz = layer.forward(ma, Sa)
                out = layer.out_features
                mu_Z = mz.reshape(-1, out).mean(dim=0)
                S_Z = Sz.reshape(-1, out).mean(dim=0)

                mu_S_t, var_S_t, mu_S2_t, var_S2_t = _layer_targets(
                    out, sigma_m, sigma_z
                )
                mu_Z_post, S_Z_post = _s_projection(
                    mu_Z, S_Z, mu_S_t, var_S_t, eps
                )
                mu_Z_post, S_Z_post = _s2_projection(
                    mu_Z_post, S_Z_post, mu_S2_t, var_S2_t, eps
                )
                _decoupled_inverse(layer, mu_Z, S_Z, mu_Z_post, S_Z_post, eps)

                ma, Sa = layer.forward(ma, Sa)
            else:
                ma, Sa = layer.forward(ma, Sa)
    net.train()
