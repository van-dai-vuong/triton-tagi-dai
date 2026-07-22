"""
Bayesian Embedding layer for TAGI.

Maps integer category indices to Gaussian-distributed vector representations.
The embedding matrix is learnable; each row i holds the mean/variance of the
Gaussian distribution for category i.

Forward (per sample b, per position t):
    cat = ma[b, t]                          (integer index)
    mz[b, t, k]  = mu_w[cat, k]
    Sz[b, t, k]  = var_w[cat, k]

If cat == padding_idx, the row is emitted as (0, 0) and no gradient flows to
mu_w[padding_idx]. Variance of the input ma is not used (the lookup is
deterministic in cat).

Backward (accumulates into delta_mu_w / delta_var_w):
    delta_mu_w [cat, k] += delta_mu[b, t, k]  * var_w[cat, k]
    delta_var_w[cat, k] += delta_var[b, t, k] * var_w[cat, k]^2

Multiple (b, t) positions with the same cat accumulate additively
(scatter_add).

Reference: cuTAGI src/embedding_cpu.cpp (feat/attn-debug).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..base import LearnableLayer
from ..param_init import gaussian_param_init
from ..update.parameters import update_parameters


class Embedding(LearnableLayer):
    """Learnable Gaussian embedding lookup.

    Parameters
    ----------
    num_embeddings : int    vocabulary size
    embedding_dim  : int    output feature dimension
    input_size     : int    sequence length (informational; no effect on compute)
    scale          : float  std-dev for mu_w initialization; var_w = scale^2
    padding_idx    : int    if a position has this index its row is zero and
                            no gradient accumulates (default -1 disables)
    device         : str or torch.device
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        input_size: int = 0,
        scale: float = 1.0,
        padding_idx: int = -1,
        device: str = "cpu",
    ) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.scale = scale
        self.padding_idx = padding_idx
        self.device = torch.device(device)
        self.has_bias = False

        # Init: mu_w ~ N(0, scale), var_w = scale^2.
        # Matches cuTAGI's `initialize_embedding_values` (gain fixed to 1).
        self.mw, self.Sw = gaussian_param_init(
            scale, 1.0, (num_embeddings, embedding_dim), device=self.device
        )
        self.mb = None
        self.Sb = None

        self.cat_idx: Tensor | None = None

        self.delta_mw: Tensor | None = None
        self.delta_Sw: Tensor | None = None
        self.delta_mb = None
        self.delta_Sb = None

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]:
        """Lookup embeddings for the integer indices in ``ma``.

        Accepts ``ma`` of shape ``(B, S)`` or ``(B, S, 1)`` (last dim squeezed).
        Returns ``(B, S, embedding_dim)``.
        """
        if ma.dim() == 3 and ma.shape[-1] == 1:
            ma = ma.squeeze(-1)
        cat = ma.long()
        self.cat_idx = cat

        mz = self.mw[cat]
        Sz = self.Sw[cat]

        if self.padding_idx >= 0:
            mask = (cat == self.padding_idx).unsqueeze(-1)
            mz = mz.masked_fill(mask, 0.0)
            Sz = Sz.masked_fill(mask, 0.0)

        return mz, Sz

    # ------------------------------------------------------------------
    #  Backward
    # ------------------------------------------------------------------
    def backward(self, delta_mz: Tensor, delta_Sz: Tensor) -> tuple[Tensor, Tensor]:
        """Scatter-add deltas into the embedding matrix.

        Matches cuTAGI's ``bwd_emb``:
            delta_mu_w [cat, k] += delta_mu[b, t, k]  * var_w[cat, k]
            delta_var_w[cat, k] += delta_var[b, t, k] * var_w[cat, k]^2

        No useful delta is propagated to the input (the index is discrete),
        so the returned tensors are zeros of matching shape.
        """
        cat = self.cat_idx
        B, S = cat.shape
        D = self.embedding_dim

        # Flatten sample and sequence axes so we can scatter into (V, D).
        cat_flat = cat.reshape(-1)
        dm_flat = delta_mz.reshape(-1, D)
        dS_flat = delta_Sz.reshape(-1, D)

        var_rows = self.Sw[cat_flat]
        dm_weighted = dm_flat * var_rows
        dS_weighted = dS_flat * var_rows * var_rows

        if self.padding_idx >= 0:
            keep = (cat_flat != self.padding_idx).unsqueeze(-1).float()
            dm_weighted = dm_weighted * keep
            dS_weighted = dS_weighted * keep

        delta_mw = torch.zeros_like(self.mw)
        delta_Sw = torch.zeros_like(self.Sw)
        idx_expanded = cat_flat.unsqueeze(-1).expand(-1, D)
        delta_mw.scatter_add_(0, idx_expanded, dm_weighted)
        delta_Sw.scatter_add_(0, idx_expanded, dS_weighted)

        self.delta_mw = delta_mw
        self.delta_Sw = delta_Sw

        delta_ma = torch.zeros_like(cat, dtype=delta_mz.dtype)
        delta_Sa = torch.zeros_like(cat, dtype=delta_Sz.dtype)
        return delta_ma, delta_Sa

    # ------------------------------------------------------------------
    #  Update
    # ------------------------------------------------------------------
    def update(self, cap_factor: float) -> None:
        update_parameters(self.mw, self.Sw, self.delta_mw, self.delta_Sw, cap_factor)

    @property
    def num_parameters(self) -> int:
        return 2 * self.mw.numel()

    def __repr__(self) -> str:
        return (
            f"Embedding(num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"padding_idx={self.padding_idx}, scale={self.scale})"
        )
