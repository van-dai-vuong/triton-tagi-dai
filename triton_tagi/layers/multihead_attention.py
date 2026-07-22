"""
Bayesian Multi-Head Self-Attention (V2) for TAGI.

Implements the ``MultiheadAttentionV2`` variant from cuTAGI's feat/attn-debug
branch: three separate Q, K, V Linear projections (rather than a single fused
3E projection as in V1). The scores pass through Remax (not Softmax); the
Jacobian from Remax is baked into the delta_query / delta_key expressions.

This port covers the configuration used by ``examples/reverse_predictor.py``:
    * ``pos_emb=""`` (no RoPE)
    * ``use_causal_mask=False``
    * ``num_kv_heads = num_heads`` (standard multi-head, no GQA)
    * ``bias`` optional; default False in the example

RoPE, causal masking, and grouped-query attention (num_kv_heads < num_heads)
are intentionally omitted; they can be added later without changing this file's
core flow.

Tensor layouts
--------------
    input / output       : (B, S, E)
    Q/K/V after reshape  : (B, H, S, D)         with E = H * D
    attention scores     : (B, H, S, S)

Forward
-------
    μ_{qk}[b, h, i, j]   = scale * Σ_m μ_q[b,h,i,m] · μ_k[b,h,j,m]
    var_{qk}[...]        = scale² * Σ_m ( S_q·S_k + S_q·μ_k² + S_k·μ_q² )
    μ_score, var_score, jcb = Remax(μ_{qk}, var_{qk}) over last dim (size S)
    μ_sv[b,h,i,m]        = Σ_j μ_score[b,h,i,j] · μ_v[b,h,j,m]
    var_sv[...]          = Σ_j ( var_score·var_v + var_score·μ_v² + μ_score²·var_v )

Backward
--------
    δμ_v[b,h,k,m] = Σ_l μ_score[b,h,l,k] · δμ_out[b,h,l,m]
    δvar_v[...]   = Σ_l μ_score²        · δvar_out
    δμ_score[b,h,k,l] = Σ_m μ_v[b,h,l,m] · δμ_out[b,h,k,m]
    δvar_score[...]   = Σ_m μ_v²        · δvar_out

After scaling by the Remax Jacobian  (linear in μ, quadratic in var):
    δμ_q = scale · Σ_l μ_k[...,l,m] · (δμ_score · jcb)[...,k,l]
    δμ_k = scale · Σ_p μ_q[...,p,m] · (δμ_score · jcb)[...,p,k]
(variance forms use μ_k², μ_q², jcb²).

The three projection deltas are then run through their respective ``Linear``
backward passes and summed to produce the input-space delta.

Reference: cuTAGI src/attention.cpp (feat/attn-debug).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..base import LearnableLayer
from ..kernels.attention import bmm_shared_left, bmm_shared_right, bmm_tagi_var
from .linear import Linear
from .remax import triton_remax


class MultiheadAttentionV2(LearnableLayer):
    """Bayesian multi-head self-attention with separate Q/K/V projections.

    Parameters
    ----------
    embed_dim       : int    E; must be divisible by ``num_heads``
    num_heads       : int    H
    seq_len         : int    S (max sequence length the layer will see)
    num_kv_heads    : int    defaults to ``num_heads``; grouped-query
                             attention (Hkv < H) is not yet implemented
    bias            : bool   add a bias on Q/K/V projections (default False)
    gain_weight     : float  gain multiplier for weight variance init
    gain_bias       : float  gain multiplier for bias variance init
    init_method     : str    "He" or "Xavier"
    pos_emb         : str    must be "" (RoPE not ported)
    use_causal_mask : bool   must be False (causal mask not ported)
    device          : str or torch.device
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        seq_len: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
        gain_weight: float = 1.0,
        gain_bias: float = 1.0,
        init_method: str = "He",
        pos_emb: str = "",
        use_causal_mask: bool = False,
        device: str = "cpu",
    ) -> None:
        if pos_emb != "":
            raise NotImplementedError(
                f"MultiheadAttentionV2 only supports pos_emb='' (got {pos_emb!r}); "
                "RoPE is not ported yet."
            )
        if use_causal_mask:
            raise NotImplementedError(
                "MultiheadAttentionV2 does not support causal masking yet."
            )
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_kv_heads != num_heads:
            raise NotImplementedError(
                "Grouped-query attention (num_kv_heads != num_heads) is not yet implemented."
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.seq_len = seq_len
        self.has_bias = bias
        self.init_method = init_method
        self.gain_weight = gain_weight
        self.gain_bias = gain_bias
        self.device = torch.device(device)

        self.head_dim = embed_dim // num_heads
        self.q_output_size = num_heads * self.head_dim
        self.k_output_size = num_kv_heads * self.head_dim
        self.v_output_size = num_kv_heads * self.head_dim

        self.q_proj = Linear(
            embed_dim, self.q_output_size, device=device,
            init_method=init_method, gain_w=gain_weight, gain_b=gain_bias, bias=bias,
        )
        self.k_proj = Linear(
            embed_dim, self.k_output_size, device=device,
            init_method=init_method, gain_w=gain_weight, gain_b=gain_bias, bias=bias,
        )
        self.v_proj = Linear(
            embed_dim, self.v_output_size, device=device,
            init_method=init_method, gain_w=gain_weight, gain_b=gain_bias, bias=bias,
        )

        self._scale = 1.0 / math.sqrt(self.head_dim)

        self._cache: dict | None = None
        self._attn_mu: Tensor | None = None
        self._attn_var: Tensor | None = None

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, ma: Tensor, Sa: Tensor) -> tuple[Tensor, Tensor]:
        """Attention forward pass. Expects ``ma`` of shape ``(B, S, E)``."""
        B, S, E = ma.shape
        if E != self.embed_dim:
            raise ValueError(f"Expected last dim {self.embed_dim}, got {E}")
        H, D = self.num_heads, self.head_dim

        # --- Q, K, V linear projections: (B, S, E) -> (B, S, H*D) ---
        mu_q_proj, var_q_proj = self.q_proj.forward(ma, Sa)
        mu_k_proj, var_k_proj = self.k_proj.forward(ma, Sa)
        mu_v_proj, var_v_proj = self.v_proj.forward(ma, Sa)

        # --- Reshape to heads: (B, S, H, D) -> (B, H, S, D) ---
        mu_q = mu_q_proj.reshape(B, S, H, D).transpose(1, 2).contiguous()
        var_q = var_q_proj.reshape(B, S, H, D).transpose(1, 2).contiguous()
        mu_k = mu_k_proj.reshape(B, S, H, D).transpose(1, 2).contiguous()
        var_k = var_k_proj.reshape(B, S, H, D).transpose(1, 2).contiguous()
        mu_v = mu_v_proj.reshape(B, S, H, D).transpose(1, 2).contiguous()
        var_v = var_v_proj.reshape(B, S, H, D).transpose(1, 2).contiguous()

        # --- Q Kᵀ (scaled TAGI product), shape (B, H, S, S) ---
        # Mean uses cuBLAS matmul; variance goes through a fused Triton
        # kernel that does both (var_a · (μ_b² + var_b)) and (μ_a² · var_b)
        # in a single pass over Q and K tiles.
        mu_k_t = mu_k.transpose(-1, -2)
        var_k_t = var_k.transpose(-1, -2)
        mu_qk = self._scale * torch.matmul(mu_q, mu_k_t)
        var_qk = bmm_tagi_var(mu_q, var_q, mu_k_t, var_k_t, scale_sq=self._scale * self._scale)

        # --- Remax over last dim (size S); flatten (B, H, S) as the row axis ---
        mu_qk_flat = mu_qk.reshape(B * H * S, S)
        var_qk_flat = var_qk.reshape(B * H * S, S)
        mu_score_flat, var_score_flat, jcb_flat = triton_remax(mu_qk_flat, var_qk_flat)
        mu_score = mu_score_flat.reshape(B, H, S, S)
        var_score = var_score_flat.reshape(B, H, S, S)
        jcb = jcb_flat.reshape(B, H, S, S)

        # --- Score @ V (TAGI product), shape (B, H, S, D) ---
        mu_sv = torch.matmul(mu_score, mu_v)
        var_sv = bmm_tagi_var(mu_score, var_score, mu_v, var_v, scale_sq=1.0)

        # --- Project output: (B, H, S, D) -> (B, S, H*D) ---
        mu_out = mu_sv.transpose(1, 2).contiguous().reshape(B, S, H * D)
        var_out = var_sv.transpose(1, 2).contiguous().reshape(B, S, H * D)

        self._cache = dict(
            B=B, S=S, H=H, D=D,
            mu_q=mu_q, mu_k=mu_k, mu_v=mu_v,
            mu_score=mu_score, jcb=jcb,
        )
        self._attn_mu = mu_score
        self._attn_var = var_score

        return mu_out, var_out

    # ------------------------------------------------------------------
    #  Backward
    # ------------------------------------------------------------------
    def backward(self, delta_mz: Tensor, delta_Sz: Tensor) -> tuple[Tensor, Tensor]:
        """Backward pass. Expects ``delta_mz`` of shape ``(B, S, E)``."""
        c = self._cache
        B, S, H, D = c["B"], c["S"], c["H"], c["D"]
        scale = self._scale

        # --- Unproject: (B, S, E) -> (B, H, S, D) ---
        d_mu = delta_mz.reshape(B, S, H, D).transpose(1, 2).contiguous()
        d_var = delta_Sz.reshape(B, S, H, D).transpose(1, 2).contiguous()

        mu_score = c["mu_score"]  # (B, H, S, S)
        mu_v = c["mu_v"]          # (B, H, S, D)
        mu_q = c["mu_q"]
        mu_k = c["mu_k"]
        jcb = c["jcb"]

        # Each of the four reductions below has a shared deterministic
        # operand (μ_score for δV, μ_v for δscore, μ_k for δQ, μ_q for δK).
        # ``bmm_shared_{left,right}`` fuses the mean and variance matmuls
        # over that shared operand into a single kernel pass.

        # --- ∂V from score: score is shared on the left ---
        mu_score_t = mu_score.transpose(-1, -2)
        delta_mu_v, delta_var_v = bmm_shared_left(mu_score_t, d_mu, d_var)

        # --- ∂score from V: V is shared on the right ---
        mu_v_t = mu_v.transpose(-1, -2)
        delta_mu_score, delta_var_score = bmm_shared_right(d_mu, d_var, mu_v_t)

        # --- Apply Remax Jacobian: mean × jcb, var × jcb² ---
        scaled_mu = delta_mu_score * jcb
        scaled_var = delta_var_score * jcb * jcb

        # --- ∂Q: μ_k is shared on the right; apply scale to mean, scale² to var ---
        delta_mu_q, delta_var_q = bmm_shared_right(scaled_mu, scaled_var, mu_k, scale=scale)

        # --- ∂K: μ_q is shared on the right, after transposing the scaled deltas ---
        scaled_mu_t = scaled_mu.transpose(-1, -2)
        scaled_var_t = scaled_var.transpose(-1, -2)
        delta_mu_k, delta_var_k = bmm_shared_right(scaled_mu_t, scaled_var_t, mu_q, scale=scale)

        # --- Reshape heads back to projection layout: (B, H, S, D) -> (B, S, H*D) ---
        dq = delta_mu_q.transpose(1, 2).contiguous().reshape(B, S, H * D)
        dSq = delta_var_q.transpose(1, 2).contiguous().reshape(B, S, H * D)
        dk = delta_mu_k.transpose(1, 2).contiguous().reshape(B, S, H * D)
        dSk = delta_var_k.transpose(1, 2).contiguous().reshape(B, S, H * D)
        dv = delta_mu_v.transpose(1, 2).contiguous().reshape(B, S, H * D)
        dSv = delta_var_v.transpose(1, 2).contiguous().reshape(B, S, H * D)

        # --- Linear backward for each projection; sum input-space deltas ---
        d_ma_q, d_Sa_q = self.q_proj.backward(dq, dSq)
        d_ma_k, d_Sa_k = self.k_proj.backward(dk, dSk)
        d_ma_v, d_Sa_v = self.v_proj.backward(dv, dSv)

        delta_ma = d_ma_q + d_ma_k + d_ma_v
        delta_Sa = d_Sa_q + d_Sa_k + d_Sa_v
        return delta_ma, delta_Sa

    # ------------------------------------------------------------------
    #  Update
    # ------------------------------------------------------------------
    def update(self, cap_factor: float) -> None:
        self.q_proj.update(cap_factor)
        self.k_proj.update(cap_factor)
        self.v_proj.update(cap_factor)

    @property
    def num_parameters(self) -> int:
        return self.q_proj.num_parameters + self.k_proj.num_parameters + self.v_proj.num_parameters

    def get_attention_scores(self) -> tuple[Tensor, Tensor]:
        """Return the most recently computed attention scores ``(μ, var)``,
        both of shape ``(B, H, S, S)``. Raises if ``forward`` has not run."""
        if self._attn_mu is None:
            raise RuntimeError("No attention scores yet; call forward first.")
        return self._attn_mu, self._attn_var

    def __repr__(self) -> str:
        return (
            f"MultiheadAttentionV2(embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, seq_len={self.seq_len}, "
            f"bias={self.has_bias})"
        )
