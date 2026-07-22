"""
Benchmark: triton-tagi attention kernels vs torch baseline.

Measures the per-call cost of each attention kernel against the equivalent
two-matmul torch expression (the math the kernels were written to fuse).

Shapes covered:
  * The two ``bmm_tagi_var`` call sites in MultiheadAttentionV2.forward.
  * The four ``bmm_shared_*`` call sites in MultiheadAttentionV2.backward.
  * A small (B*H, M, L, K) sweep to locate the Triton/torch crossover.

The default shapes match ``examples/reverse_predictor.py`` (B=64, H=1, S=8, D=32).

Usage:
    cd /home/mf2/triton
    source /home/mf2/.miniconda3/etc/profile.d/conda.sh && conda activate cuTAGI
    python benchmarks/bench_attention.py
"""

from __future__ import annotations

import torch

from triton_tagi.kernels.attention import (
    bmm_shared_left,
    bmm_shared_right,
    bmm_tagi_var,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 50
REPS = 200


def _cuda_time_us(fn) -> float:
    """Median device-time in microseconds, using CUDA events for <100µs kernels."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(REPS)]
    for i in range(REPS):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
    times.sort()
    return times[len(times) // 2]


# ── Torch baselines (the math the Triton kernels fuse) ─────────────────────────


def torch_bmm_tagi_var(mu_a, var_a, mu_b, var_b, scale_sq=1.0):
    """Two-matmul TAGI variance: var_a·(μ_b² + var_b) + μ_a²·var_b."""
    out = torch.matmul(var_a, mu_b * mu_b + var_b) + torch.matmul(mu_a * mu_a, var_b)
    return out.mul_(scale_sq) if scale_sq != 1.0 else out


def torch_bmm_shared_right(a_mean, a_var, b, scale=1.0):
    """Two matmuls with a shared deterministic right operand."""
    mean = torch.matmul(a_mean, b)
    var = torch.matmul(a_var, b * b)
    if scale != 1.0:
        mean.mul_(scale)
        var.mul_(scale * scale)
    return mean, var


def torch_bmm_shared_left(a, b_mean, b_var, scale=1.0):
    mean = torch.matmul(a, b_mean)
    var = torch.matmul(a * a, b_var)
    if scale != 1.0:
        mean.mul_(scale)
        var.mul_(scale * scale)
    return mean, var


# ── Benchmark drivers ──────────────────────────────────────────────────────────


def _rand(shape):
    return torch.randn(*shape, device=DEVICE)


def _pos(shape):
    return torch.rand(*shape, device=DEVICE).abs() * 0.1 + 1e-3


def bench_tagi_var(batch, M, K, L, scale_sq=1.0):
    """Time bmm_tagi_var vs torch baseline. Inputs shaped (batch, M, K) × (batch, K, L)."""
    mu_a, var_a = _rand((batch, M, K)), _pos((batch, M, K))
    mu_b, var_b = _rand((batch, K, L)), _pos((batch, K, L))

    triton_us = _cuda_time_us(lambda: bmm_tagi_var(mu_a, var_a, mu_b, var_b, scale_sq=scale_sq))
    torch_us = _cuda_time_us(lambda: torch_bmm_tagi_var(mu_a, var_a, mu_b, var_b, scale_sq))
    return triton_us, torch_us


def bench_shared_right(batch, M, K, L, scale=1.0):
    a_mean, a_var = _rand((batch, M, K)), _pos((batch, M, K))
    b = _rand((batch, K, L))

    triton_us = _cuda_time_us(lambda: bmm_shared_right(a_mean, a_var, b, scale=scale))
    torch_us = _cuda_time_us(lambda: torch_bmm_shared_right(a_mean, a_var, b, scale))
    return triton_us, torch_us


def bench_shared_left(batch, M, K, L, scale=1.0):
    a = _rand((batch, M, K))
    b_mean, b_var = _rand((batch, K, L)), _pos((batch, K, L))

    triton_us = _cuda_time_us(lambda: bmm_shared_left(a, b_mean, b_var, scale=scale))
    torch_us = _cuda_time_us(lambda: torch_bmm_shared_left(a, b_mean, b_var, scale))
    return triton_us, torch_us


# ── Reporting ──────────────────────────────────────────────────────────────────


def _row(label, batch, M, K, L, triton_us, torch_us):
    speedup = torch_us / triton_us
    marker = "✓" if speedup >= 1.0 else "✗"
    print(
        f"  {label:<32}  bN={batch:>5}  M={M:>3} K={K:>3} L={L:>3}  "
        f"triton={triton_us:7.1f}µs  torch={torch_us:7.1f}µs  "
        f"speedup={speedup:5.2f}× {marker}"
    )


def bench_reverse_predictor_shapes():
    """Exact call sites from MultiheadAttentionV2 at reverse_predictor defaults."""
    B, H, S, D = 64, 1, 8, 32
    BH = B * H
    scale = 1.0 / (D**0.5)
    scale_sq = scale * scale

    print(f"\n## reverse_predictor shapes (B={B}, H={H}, S={S}, D={D})")
    print(f"   batch dim folded into N=B*H={BH}\n")

    # Forward
    print("  -- forward --")
    t, p = bench_tagi_var(BH, S, D, S, scale_sq=scale_sq)
    _row("bmm_tagi_var (Q @ K^T)", BH, S, D, S, t, p)
    t, p = bench_tagi_var(BH, S, S, D, scale_sq=1.0)
    _row("bmm_tagi_var (Score @ V)", BH, S, S, D, t, p)

    # Backward
    print("  -- backward --")
    t, p = bench_shared_left(BH, S, S, D)
    _row("bmm_shared_left  (∂V)", BH, S, S, D, t, p)
    t, p = bench_shared_right(BH, S, D, S)
    _row("bmm_shared_right (∂score)", BH, S, D, S, t, p)
    t, p = bench_shared_right(BH, S, S, D, scale=scale)
    _row("bmm_shared_right (∂Q)", BH, S, S, D, t, p)
    t, p = bench_shared_right(BH, S, S, D, scale=scale)
    _row("bmm_shared_right (∂K)", BH, S, S, D, t, p)


def bench_size_sweep():
    """Crossover sweep: vary batch and seq_len holding head_dim=32."""
    print("\n## size sweep — bmm_tagi_var (square M=L=S, K=D=32)\n")
    D = 32
    for batch in (16, 64, 256, 1024):
        for S in (8, 16, 32, 64, 128):
            t, p = bench_tagi_var(batch, S, D, S)
            _row(f"S={S}", batch, S, D, S, t, p)
        print()


def bench_full_layer():
    """End-to-end MultiheadAttentionV2.forward+backward vs a pure-torch reference
    that computes the same TAGI math with plain torch.matmul + elementwise ops.

    The pure-torch reference uses the same Linear layers from triton-tagi for Q/K/V
    so we isolate the attention core (matmuls + variance + remax) from the
    projection path.
    """
    from triton_tagi.layers import Linear, MultiheadAttentionV2
    from triton_tagi.layers.remax import triton_remax

    B, S, E, H = 64, 8, 32, 1
    D = E // H
    scale = 1.0 / (D**0.5)
    scale_sq = scale * scale

    mha = MultiheadAttentionV2(E, H, S, bias=False, device=DEVICE)

    # Pure-torch reference: reuse q/k/v projections from the triton-tagi layer,
    # but compute attention core with plain torch ops (two matmuls for every var).
    q_proj, k_proj, v_proj = mha.q_proj, mha.k_proj, mha.v_proj

    def torch_attn_forward(ma, Sa):
        mu_q, var_q = q_proj.forward(ma, Sa)
        mu_k, var_k = k_proj.forward(ma, Sa)
        mu_v, var_v = v_proj.forward(ma, Sa)
        mu_q = mu_q.reshape(B, S, H, D).transpose(1, 2)
        var_q = var_q.reshape(B, S, H, D).transpose(1, 2)
        mu_k = mu_k.reshape(B, S, H, D).transpose(1, 2)
        var_k = var_k.reshape(B, S, H, D).transpose(1, 2)
        mu_v = mu_v.reshape(B, S, H, D).transpose(1, 2)
        var_v = var_v.reshape(B, S, H, D).transpose(1, 2)
        mu_k_t = mu_k.transpose(-1, -2)
        var_k_t = var_k.transpose(-1, -2)
        mu_qk = scale * torch.matmul(mu_q, mu_k_t)
        var_qk = scale_sq * (
            torch.matmul(var_q, mu_k_t * mu_k_t + var_k_t)
            + torch.matmul(mu_q * mu_q, var_k_t)
        )
        mu_s, var_s, jcb = triton_remax(mu_qk.reshape(-1, S), var_qk.reshape(-1, S))
        mu_s = mu_s.reshape(B, H, S, S)
        var_s = var_s.reshape(B, H, S, S)
        mu_sv = torch.matmul(mu_s, mu_v)
        var_sv = torch.matmul(var_s, mu_v * mu_v + var_v) + torch.matmul(mu_s * mu_s, var_v)
        return mu_sv.transpose(1, 2).reshape(B, S, E), var_sv.transpose(1, 2).reshape(B, S, E)

    ma = torch.randn(B, S, E, device=DEVICE)
    Sa = torch.rand(B, S, E, device=DEVICE).abs() * 0.1
    dma = torch.randn(B, S, E, device=DEVICE)
    dSa = torch.rand(B, S, E, device=DEVICE).abs() * 0.01

    def triton_fwd_bwd():
        mha.forward(ma, Sa)
        mha.backward(dma, dSa)

    # Prime the cache: forward+backward once so `_cache` is populated.
    mha.forward(ma, Sa)
    mha.backward(dma, dSa)

    def triton_only_bwd():
        mha.backward(dma, dSa)

    # For the pure-torch version, we only time the attention core (fwd) since
    # implementing the backward in pure torch is out of scope for this check.
    t_fwd_bwd = _cuda_time_us(triton_fwd_bwd)
    t_fwd = _cuda_time_us(lambda: mha.forward(ma, Sa))
    t_bwd = _cuda_time_us(triton_only_bwd)
    t_torch_fwd = _cuda_time_us(lambda: torch_attn_forward(ma, Sa))

    print(f"\n## full-layer end-to-end (B={B}, S={S}, E={E}, H={H})\n")
    print(f"  MHA.forward  (triton kernels)       : {t_fwd:7.1f}µs")
    print(f"  MHA.forward  (pure torch math)      : {t_torch_fwd:7.1f}µs  "
          f"[{t_torch_fwd / t_fwd:.2f}× vs triton]")
    print(f"  MHA.backward (triton kernels)       : {t_bwd:7.1f}µs")
    print(f"  MHA.forward + backward (triton)     : {t_fwd_bwd:7.1f}µs")


def main():
    print("=" * 90)
    print(f"  Attention kernel benchmark — {torch.cuda.get_device_name(0)}")
    print(f"  warmup={WARMUP}, reps={REPS} (median, CUDA events)")
    print("=" * 90)

    bench_reverse_predictor_shapes()
    bench_full_layer()
    bench_size_sweep()


if __name__ == "__main__":
    main()
