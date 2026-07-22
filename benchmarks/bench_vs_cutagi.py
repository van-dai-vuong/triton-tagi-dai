"""
Benchmark: triton-tagi vs cuTAGI — wall-clock time per forward+backward+update step.

Benchmarks:
  1. Linear (standalone layer)    — Conv2d.backward() crashes standalone in pytagi, so
  2. Conv2D network               —   Conv2D and BN are benchmarked as full networks
  3. BatchNorm2D network          —   with a Linear head to make backward work in cuTAGI.

Batch sizes: 1, 16, 32, 64, 256, 1024

Network architectures:
  Linear:         Linear(512, 512)
  Conv2D net:     Conv2D(32,32,3,pad=1,16×16) → ReLU → Flatten → Linear(flat, 64)
  BatchNorm2D net: Conv2D(32,32,3,pad=1,16×16) → BN(32) → ReLU → Flatten → Linear(flat, 64)

Results are written to benchmarks/results.md.

Usage:
    cd /home/mf2/triton
    source /home/mf2/.miniconda3/etc/profile.d/conda.sh && conda activate cuTAGI
    python benchmarks/bench_vs_cutagi.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZES = [1, 16, 32, 64, 256, 1024]
WARMUP = 10
REPS = 50
SIGMA_V = 0.1

# ── Layer/network configs ──────────────────────────────────────────────────────

LINEAR_IN = 512
LINEAR_OUT = 512

CONV_C_IN = 32
CONV_C_OUT = 32
CONV_K = 3
CONV_H = 16
CONV_W = 16
CONV_FLAT = CONV_C_OUT * CONV_H * CONV_W  # padding=1 preserves spatial
CONV_HEAD_OUT = 64


# ── Timing helper ──────────────────────────────────────────────────────────────


def _cuda_time_ms(fn, warmup: int, reps: int) -> float:
    """Return median wall-clock time in milliseconds over `reps` runs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return times[len(times) // 2]


# ── triton-tagi benchmarks ─────────────────────────────────────────────────────


def bench_triton_linear(batch: int) -> float:
    from triton_tagi.layers import Linear
    from triton_tagi.update.parameters import get_cap_factor

    layer = Linear(LINEAR_IN, LINEAR_OUT, device=DEVICE)
    ma = torch.randn(batch, LINEAR_IN, device=DEVICE)
    Sa = torch.rand(batch, LINEAR_IN, device=DEVICE).abs() * 0.1
    dma = torch.randn(batch, LINEAR_OUT, device=DEVICE)
    dSa = torch.rand(batch, LINEAR_OUT, device=DEVICE).abs() * 0.01
    cap = get_cap_factor(batch)

    def step():
        layer.forward(ma, Sa)
        layer.backward(dma, dSa)
        layer.update(cap)

    return _cuda_time_ms(step, WARMUP, REPS)


def _build_triton_conv2d_net(with_bn: bool):
    """Conv2D(→BN)→ReLU→Flatten→Linear as a triton-tagi Sequential."""
    from triton_tagi.layers import BatchNorm2D, Conv2D, Flatten, Linear, ReLU
    from triton_tagi.network import Sequential

    layers = [Conv2D(CONV_C_IN, CONV_C_OUT, CONV_K, padding=1, device=DEVICE)]
    if with_bn:
        layers.append(BatchNorm2D(CONV_C_OUT, device=DEVICE))
    layers += [ReLU(), Flatten(), Linear(CONV_FLAT, CONV_HEAD_OUT, device=DEVICE)]
    return Sequential(layers, device=DEVICE)


def bench_triton_conv2d(batch: int) -> float:
    net = _build_triton_conv2d_net(with_bn=False)
    net.train()
    x = torch.randn(batch, CONV_C_IN, CONV_H, CONV_W, device=DEVICE)
    y = torch.randn(batch, CONV_HEAD_OUT, device=DEVICE)

    return _cuda_time_ms(lambda: net.step(x, y, SIGMA_V), WARMUP, REPS)


def bench_triton_batchnorm2d(batch: int) -> float:
    net = _build_triton_conv2d_net(with_bn=True)
    net.train()
    x = torch.randn(batch, CONV_C_IN, CONV_H, CONV_W, device=DEVICE)
    y = torch.randn(batch, CONV_HEAD_OUT, device=DEVICE)

    return _cuda_time_ms(lambda: net.step(x, y, SIGMA_V), WARMUP, REPS)


# ── cuTAGI benchmarks ──────────────────────────────────────────────────────────


def _pytagi_available() -> bool:
    try:
        import pytagi.nn  # noqa: F401
        return True
    except ImportError:
        return False


def _cutagi_step(net, ma_np: np.ndarray, Sa_np: np.ndarray, y_np: np.ndarray) -> None:
    from pytagi.nn import OutputUpdater

    updater = OutputUpdater(net.device)
    net.forward(ma_np, Sa_np)
    updater.update(
        output_states=net.output_z_buffer,
        mu_obs=y_np,
        var_obs=np.full_like(y_np, SIGMA_V**2),
        delta_states=net.input_delta_z_buffer,
    )
    net.backward()
    net.step()


def bench_cutagi_linear(batch: int) -> float | None:
    if not _pytagi_available():
        return None
    import pytagi.nn as nn

    net = nn.Sequential(nn.Linear(LINEAR_IN, LINEAR_OUT))
    net.to_device("cuda")

    rng = np.random.default_rng(0)
    ma_np = rng.standard_normal(batch * LINEAR_IN).astype(np.float32)
    Sa_np = (rng.random(batch * LINEAR_IN) * 0.1 + 1e-6).astype(np.float32)
    y_np = rng.standard_normal(batch * LINEAR_OUT).astype(np.float32)

    return _cuda_time_ms(lambda: _cutagi_step(net, ma_np, Sa_np, y_np), WARMUP, REPS)


def _build_cutagi_conv2d_net(with_bn: bool):
    import pytagi.nn as nn

    layers = [
        nn.Conv2d(CONV_C_IN, CONV_C_OUT, CONV_K, padding=1, in_width=CONV_W, in_height=CONV_H),
    ]
    if with_bn:
        layers.append(nn.BatchNorm2d(CONV_C_OUT))
    layers += [nn.ReLU(), nn.Linear(CONV_FLAT, CONV_HEAD_OUT)]
    net = nn.Sequential(*layers)
    net.to_device("cuda")
    return net


def bench_cutagi_conv2d(batch: int) -> float | None:
    if not _pytagi_available():
        return None

    net = _build_cutagi_conv2d_net(with_bn=False)
    rng = np.random.default_rng(0)
    ma_np = rng.standard_normal(batch * CONV_C_IN * CONV_H * CONV_W).astype(np.float32)
    Sa_np = (rng.random(batch * CONV_C_IN * CONV_H * CONV_W) * 0.1 + 1e-6).astype(np.float32)
    y_np = rng.standard_normal(batch * CONV_HEAD_OUT).astype(np.float32)

    return _cuda_time_ms(lambda: _cutagi_step(net, ma_np, Sa_np, y_np), WARMUP, REPS)


def bench_cutagi_batchnorm2d(batch: int) -> float | None:
    if not _pytagi_available():
        return None

    net = _build_cutagi_conv2d_net(with_bn=True)
    rng = np.random.default_rng(0)
    ma_np = rng.standard_normal(batch * CONV_C_IN * CONV_H * CONV_W).astype(np.float32)
    Sa_np = (rng.random(batch * CONV_C_IN * CONV_H * CONV_W) * 0.1 + 1e-6).astype(np.float32)
    y_np = rng.standard_normal(batch * CONV_HEAD_OUT).astype(np.float32)

    return _cuda_time_ms(lambda: _cutagi_step(net, ma_np, Sa_np, y_np), WARMUP, REPS)


# ── Table formatting ───────────────────────────────────────────────────────────


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def _speedup(triton_ms: float, cutagi_ms: float | None) -> str:
    if cutagi_ms is None:
        return "N/A"
    return f"{cutagi_ms / triton_ms:.2f}×"


def _md_table(label: str, note: str, rows: list[dict]) -> str:
    lines = [
        f"### {label}",
        "",
        f"_{note}_",
        "",
        "| Batch | triton-tagi (ms) | cuTAGI (ms) | Speedup |",
        "|------:|----------------:|------------:|--------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['batch']:>5} "
            f"| {_fmt(r['triton']):>15} "
            f"| {_fmt(r['cutagi']):>11} "
            f"| {_speedup(r['triton'], r['cutagi']):>7} |"
        )
    lines.append("")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

BENCHMARKS = [
    (
        "Linear",
        "Standalone layer: Linear(512, 512). Triton-tagi times the layer directly; "
        "cuTAGI wraps it in a 1-layer Sequential.",
        bench_triton_linear,
        bench_cutagi_linear,
    ),
    (
        "Conv2D network",
        "Full network: Conv2D(32,32,3,pad=1,16×16) → ReLU → Flatten → Linear(8192,64). "
        "Conv2D standalone backward crashes in pytagi, so both sides time the full network.",
        bench_triton_conv2d,
        bench_cutagi_conv2d,
    ),
    (
        "BatchNorm2D network",
        "Full network: Conv2D(32,32,3,pad=1,16×16) → BN(32) → ReLU → Flatten → Linear(8192,64). "
        "pytagi BatchNorm2d requires a preceding Conv2d, so both sides time the full network.",
        bench_triton_batchnorm2d,
        bench_cutagi_batchnorm2d,
    ),
]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for benchmarking.")

    has_cutagi = _pytagi_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"cuTAGI available: {has_cutagi}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Warmup={WARMUP}, reps={REPS}, metric=median\n")

    all_tables: list[str] = []

    for label, note, triton_fn, cutagi_fn in BENCHMARKS:
        print(f"Benchmarking {label}...")
        rows = []
        for batch in BATCH_SIZES:
            t_ms = triton_fn(batch)
            c_ms = cutagi_fn(batch) if has_cutagi else None
            rows.append({"batch": batch, "triton": t_ms, "cutagi": c_ms})
            sp = _speedup(t_ms, c_ms)
            print(f"  batch={batch:>4}: triton={t_ms:.3f}ms  cutagi={_fmt(c_ms)}ms  speedup={sp}")
        all_tables.append(_md_table(label, note, rows))
        print()

    gpu_name = torch.cuda.get_device_name(0)
    header = (
        "# Benchmark Results: triton-tagi vs cuTAGI\n\n"
        f"**GPU:** {gpu_name}  \n"
        f"**Metric:** median wall-clock time over {REPS} runs (ms), "
        f"{WARMUP} warmup iterations  \n"
        "**Step:** forward + backward + update  \n\n"
        "---\n\n"
    )

    out_path = Path(__file__).parent / "results.md"
    out_path.write_text(header + "\n".join(all_tables))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
