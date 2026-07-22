"""
Sinusoidal positional encoding (non-learnable, additive).

Precomputes a cache of shape ``(max_seq_len, embed_dim)`` at construction time
and adds it to the input means on forward. The backward pass is the identity
(Jacobian = 1), so deltas propagate unchanged. The variance is left unchanged
because PE is a deterministic constant.

cuTAGI frequency convention (reproduced exactly, feat/attn-debug):
    freq(d)    = 1 / 10000^(d / embed_dim)        (d in float, not d // 2 * 2)
    pe[t, d]   = sin(t * freq(d))  if d is even
                 cos(t * freq(d))  if d is odd

Reference: cuTAGI src/positional_encoding.cpp.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..base import Layer


class PositionalEncoding(Layer):
    """Fixed sinusoidal positional encoding (no trainable parameters).

    Parameters
    ----------
    embed_dim   : int    last-dim feature size the input must carry
    max_seq_len : int    upper bound on sequence length (default 2048)
    device      : str or torch.device
    """

    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int = 2048,
        device: str = "cpu",
    ) -> None:
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)

        pe = torch.empty(max_seq_len, embed_dim, device=self.device)
        d = torch.arange(embed_dim, dtype=torch.float32, device=self.device)
        pos = torch.arange(max_seq_len, dtype=torch.float32, device=self.device)
        freq = 1.0 / torch.pow(10000.0, d / embed_dim)
        angle = pos.unsqueeze(1) * freq.unsqueeze(0)
        even_mask = (torch.arange(embed_dim, device=self.device) % 2 == 0)
        pe = torch.where(even_mask.unsqueeze(0), torch.sin(angle), torch.cos(angle))
        self.pe_cache = pe

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]:
        """Add the sinusoidal cache to the input means. Expects ``ma`` of shape
        ``(B, S, embed_dim)``; broadcasts the cache across the batch."""
        S = ma.shape[-2]
        pe = self.pe_cache[:S]
        return ma + pe, Sa

    # ------------------------------------------------------------------
    #  Backward (identity)
    # ------------------------------------------------------------------
    def backward(self, delta_ma: Tensor, delta_Sa: Tensor) -> tuple[Tensor, Tensor]:
        return delta_ma, delta_Sa

    def __repr__(self) -> str:
        return f"PositionalEncoding(embed_dim={self.embed_dim}, max_seq_len={self.max_seq_len})"
