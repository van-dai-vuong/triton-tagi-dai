"""
Profiling: find the top CUDA bottlenecks in triton-tagi layers.

Profiles forward+backward+update for:
  - Linear(512, 512)
  - Conv2D network: Conv2D(32,32,3,pad=1,16x16) -> ReLU -> Flatten -> Linear(8192,64)
  - BatchNorm2D network: Conv2D -> BN(32) -> ReLU -> Flatten -> Linear(8192,64)

at batch size 256 (compute-dominated regime where kernel performance matters most).

Output:
  - Console: top-10 ops by CUDA time for each benchmark
  - benchmarks/profile_results.txt: full report

Usage:
    cd /home/mf2/triton
    source /home/mf2/.miniconda3/etc/profile.d/conda.sh && conda activate cuTAGI
    python benchmarks/profile_bottlenecks.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.profiler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 256
WARMUP = 10
ACTIVE = 20  # steps captured by profiler

LINEAR_IN = 512
LINEAR_OUT = 512
CONV_C_IN = 32
CONV_C_OUT = 32
CONV_K = 3
CONV_H = 16
CONV_W = 16
CONV_FLAT = CONV_C_OUT * CONV_H * CONV_W
CONV_HEAD_OUT = 64
SIGMA_V = 0.1


# ── Step functions ─────────────────────────────────────────────────────────────


def make_linear_step():
    from triton_tagi.layers import Linear
    from triton_tagi.update.parameters import get_cap_factor

    layer = Linear(LINEAR_IN, LINEAR_OUT, device=DEVICE)
    ma = torch.randn(BATCH, LINEAR_IN, device=DEVICE)
    Sa = torch.rand(BATCH, LINEAR_IN, device=DEVICE).abs() * 0.1
    dma = torch.randn(BATCH, LINEAR_OUT, device=DEVICE)
    dSa = torch.rand(BATCH, LINEAR_OUT, device=DEVICE).abs() * 0.01
    cap = get_cap_factor(BATCH)

    def step():
        layer.forward(ma, Sa)
        layer.backward(dma, dSa)
        layer.update(cap)

    return step


def make_conv2d_step():
    from triton_tagi.layers import Conv2D, Flatten, Linear, ReLU
    from triton_tagi.network import Sequential

    net = Sequential(
        [
            Conv2D(CONV_C_IN, CONV_C_OUT, CONV_K, padding=1, device=DEVICE),
            ReLU(),
            Flatten(),
            Linear(CONV_FLAT, CONV_HEAD_OUT, device=DEVICE),
        ],
        device=DEVICE,
    )
    net.train()
    x = torch.randn(BATCH, CONV_C_IN, CONV_H, CONV_W, device=DEVICE)
    y = torch.randn(BATCH, CONV_HEAD_OUT, device=DEVICE)

    def step():
        net.step(x, y, SIGMA_V)

    return step


def make_bn_step():
    from triton_tagi.layers import BatchNorm2D, Conv2D, Flatten, Linear, ReLU
    from triton_tagi.network import Sequential

    net = Sequential(
        [
            Conv2D(CONV_C_IN, CONV_C_OUT, CONV_K, padding=1, device=DEVICE),
            BatchNorm2D(CONV_C_OUT, device=DEVICE),
            ReLU(),
            Flatten(),
            Linear(CONV_FLAT, CONV_HEAD_OUT, device=DEVICE),
        ],
        device=DEVICE,
    )
    net.train()
    x = torch.randn(BATCH, CONV_C_IN, CONV_H, CONV_W, device=DEVICE)
    y = torch.randn(BATCH, CONV_HEAD_OUT, device=DEVICE)

    def step():
        net.step(x, y, SIGMA_V)

    return step


TARGETS = [
    ("Linear(512→512)", make_linear_step),
    ("Conv2D network", make_conv2d_step),
    ("BatchNorm2D network", make_bn_step),
]


# ── Profiling helper ───────────────────────────────────────────────────────────

def _cuda_us(e) -> float:
    return getattr(e, "self_cuda_time_total",
                   getattr(e, "self_device_time_total", 0.0))

def _cpu_us(e) -> float:
    return getattr(e, "self_cpu_time_total", 0.0)


def profile_step(step_fn) -> tuple[list, float]:
    """Run ACTIVE steps under the profiler, return (sorted_avgs, total_cuda_ms)."""
    # Warmup outside profiler so Triton JIT compilation doesn't appear
    for _ in range(WARMUP):
        step_fn()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(ACTIVE):
            step_fn()
        torch.cuda.synchronize()

    avgs = prof.key_averages()
    avgs_sorted = sorted(avgs, key=_cuda_us, reverse=True)
    total_cuda_us = sum(_cuda_us(e) for e in avgs_sorted)
    return avgs_sorted, total_cuda_us / 1000.0


def format_table(avgs_sorted: list, total_cuda_ms: float) -> str:
    header = f"{'Op':<52} {'CPU ms':>8} {'CUDA ms':>8} {'CUDA%':>6} {'Calls':>6}"
    sep = "-" * 86

    lines = [header, sep]
    for i, e in enumerate(avgs_sorted[:10]):
        cuda_ms = _cuda_us(e) / 1000.0
        cpu_ms = _cpu_us(e) / 1000.0
        pct = 100.0 * _cuda_us(e) / (total_cuda_ms * 1000.0) if total_cuda_ms > 0 else 0.0
        marker = "  ◀ TOP" if i < 3 else ""
        lines.append(
            f"{e.key:<52} {cpu_ms:>8.3f} {cuda_ms:>8.3f} {pct:>5.1f}% {e.count:>6}{marker}"
        )
    lines += [sep, f"{'Total CUDA time (self):':<52} {'':>8} {total_cuda_ms:>8.3f}", ""]
    return "\n".join(lines)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for profiling.")

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")
    print(f"Batch size: {BATCH}  |  warmup={WARMUP}, profiled steps={ACTIVE}\n")

    report_lines = [
        "# Profiling Results: triton-tagi CUDA Bottlenecks",
        "",
        f"**GPU:** {gpu}  ",
        f"**Batch size:** {BATCH}  ",
        f"**Profiled steps:** {ACTIVE} (after {WARMUP} warmup steps)  ",
        "",
        "Top-10 ops by CUDA self-time. Ops marked ◀ TOP are the three biggest bottlenecks.",
        "",
        "---",
        "",
    ]

    for label, make_fn in TARGETS:
        print(f"Profiling {label}...")
        step_fn = make_fn()
        avgs_sorted, total_cuda_ms = profile_step(step_fn)
        table = format_table(avgs_sorted, total_cuda_ms)
        print(table)
        report_lines += [f"## {label}", "", "```", table, "```", ""]

    out_path = Path(__file__).parent / "profile_results.txt"
    out_path.write_text("\n".join(report_lines))
    print(f"Full report written to {out_path}")


if __name__ == "__main__":
    main()
