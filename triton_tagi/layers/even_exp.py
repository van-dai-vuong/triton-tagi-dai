"""
EvenExp activation layer for TAGI-V (heteroscedastic noise learning).

cuTAGI's TAGI-V variance head uses an exponential activation. The regression
example builds ``SplitActivation(Exp())`` (cuTAGI ``examples/regression_heteros.py``),
applying ``Exp`` to the odd (variance) stream and identity to the even (mean)
stream. This is the triton port of that head — the exp counterpart of
:class:`EvenSoftplus`, a drop-in swap with the same interleaved 2K layout:

    - Even indices (0, 2, 4, ...): mean predictions   → identity (pass-through)
    - Odd  indices (1, 3, 5, ...): variance predictions → exp (log-normal)

For an odd-slot pre-activation ``z ~ N(μ, S)``, ``a = exp(z)`` is log-normal with
**exact** moments (no delta-method / Taylor linearisation), matching cuTAGI's
``exp_mean_var`` in ``src/activation.cpp`` for ``Exp(scale, shift)``::

    μ' = scale·μ + shift,        S' = scale²·S
    μ_a = max(exp(μ' + S'/2),               1e-6)
    S_a = max(exp(2μ' + S')·(exp(S') − 1),  1e-6)
    jcb = μ_a · scale                       (= Cov(z, a)/Var(z))

Backward is an **identity passthrough**. cuTAGI does not apply the exp Jacobian
in the activation's backward; the heteroscedastic output updater
(``update_delta_z_cuda_heteros``) folds the smoother gain ``jv = μ_a/S_a`` into
the pre-activation delta itself. The triton port mirrors this exactly: the fold
lives in :func:`triton_tagi.update.observation.compute_innovation` (heteros
branch), so EvenExp is a forward-only moment layer for the TAGI-V output head —
do not use it as a generic hidden activation.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..base import Layer

# cuTAGI exp_mean_var floors both moments at 0.000001f.
_FLOOR = 1e-6


class EvenExp(Layer):
    """Exponential variance-head activation (TAGI-V), interleaved 2K layout.

    Applies ``exp`` (exact log-normal moments) to odd-indexed positions
    (variance predictions) and passes even-indexed positions (means) through
    unchanged. Output width must be ``2 * half_width``.

    Parameters
    ----------
    half_width : int
        ``K`` = number of target dimensions (even/odd pairs).
    scale, shift : float
        Pre-exponent affine, mirroring cuTAGI ``Exp(scale, shift)``. Defaults
        (``1.0`` / ``0.0``) match the cuTAGI regression-heteros example.
    """

    def __init__(self, half_width: int, scale: float = 1.0, shift: float = 0.0) -> None:
        self.half_width = half_width
        self.scale = float(scale)
        self.shift = float(shift)

    def forward(self, mz: Tensor, Sz: Tensor) -> tuple[Tensor, Tensor]:
        odd = slice(1, None, 2)
        ma = mz.clone()
        Sa = Sz.clone()

        m = self.scale * mz[..., odd] + self.shift
        s = (self.scale * self.scale) * Sz[..., odd]

        # Exact log-normal moments, matching cuTAGI exp_mean_var bit-for-bit.
        mu_a = torch.exp(m + 0.5 * s).clamp_min(_FLOOR)
        var_a = (torch.exp(2.0 * m + s) * (torch.exp(s) - 1.0)).clamp_min(_FLOOR)

        ma[..., odd] = mu_a
        Sa[..., odd] = var_a
        return ma, Sa

    def backward(self, delta_ma: Tensor, delta_Sa: Tensor) -> tuple[Tensor, Tensor]:
        # Identity passthrough — the heteroscedastic output updater folds the
        # exp Jacobian (smoother gain) into the pre-activation delta, exactly as
        # cuTAGI's update_delta_z_cuda_heteros does (see module docstring).
        return delta_ma, delta_Sa

    def __repr__(self) -> str:
        return (
            f"EvenExp(half_width={self.half_width}, "
            f"scale={self.scale}, shift={self.shift})"
        )
