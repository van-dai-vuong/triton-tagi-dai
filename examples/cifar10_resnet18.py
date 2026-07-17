"""
CIFAR-10 Classification — ResNet-18 — triton-tagi example.

Architecture (CIFAR-10 adaptation — 3×3 stem, no max-pool):
    Stem:    Conv(3→64, 3×3, p=1) → ReLU → BN           [32×32]
    Stage 1: ResBlock(64,  64,  s=1) × 2                 [32×32]
    Stage 2: ResBlock(64,  128, s=2) + ResBlock(128, 128) [16×16]
    Stage 3: ResBlock(128, 256, s=2) + ResBlock(256, 256) [8×8]
    Stage 4: ResBlock(256, 512, s=2) + ResBlock(512, 512) [4×4]
    Head:    AvgPool(4) → Flatten → FC(512→10) → Remax

Each ResBlock: Conv(3×3,s) → ReLU → BN → Conv(3×3) → ReLU → BN + shortcut.
Projection shortcut (stride>1 or ch mismatch): Conv(2×2, s=2) → ReLU → BN.

σ_v is decayed each epoch: σ_v(t) = max(σ_v · rate^t, σ_v_min).
Matches cuTAGI's exponential_scheduler (default: 1.0 → 0.3 at ×0.95/ep).

Usage:
    python examples/cifar10_resnet18.py
    python examples/cifar10_resnet18.py --n_epochs 30
    python examples/cifar10_resnet18.py --data_dir /path/to/data --no_augment
    python examples/cifar10_resnet18.py --help
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from triton_tagi import (
    AvgPool2D,
    BatchNorm2D,
    Conv2D,
    Flatten,
    Linear,
    ReLU,
    Remax,
    ResBlock,
    Sequential,
)
from triton_tagi.checkpoint import RunDir


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(
    data_dir: str = "data",
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load CIFAR-10 as (N, 3, 32, 32) tensors on ``device``.

    Returns:
        x_train (50000,3,32,32), y_train_oh (50000,10), y_train_labels (50000,),
        x_test (10000,3,32,32), y_test_labels (10000,).
    """
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)])
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=norm)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=norm)

    x_train = torch.stack([img for img, _ in train_ds]).to(device)
    y_train = torch.tensor([lbl for _, lbl in train_ds], device=device)
    x_test = torch.stack([img for img, _ in test_ds]).to(device)
    y_test = torch.tensor([lbl for _, lbl in test_ds], device=device)

    y_train_oh = torch.zeros(len(y_train), 10, device=device)
    y_train_oh.scatter_(1, y_train.unsqueeze(1), 1.0)

    return x_train, y_train_oh, y_train, x_test, y_test


# ---------------------------------------------------------------------------
#  GPU augmentation (random horizontal flip + random crop, no CPU round-trip)
# ---------------------------------------------------------------------------

def gpu_augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random horizontal flip and random crop applied to a batch on-device."""
    B, C, H, W = x.shape
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    top = torch.randint(0, 2 * pad, (B,), device=x.device)
    left = torch.randint(0, 2 * pad, (B,), device=x.device)
    rows = top.unsqueeze(1) + torch.arange(H, device=x.device).unsqueeze(0)
    cols = left.unsqueeze(1) + torch.arange(W, device=x.device).unsqueeze(0)
    return x_pad[
        torch.arange(B, device=x.device)[:, None, None, None],
        torch.arange(C, device=x.device)[None, :, None, None],
        rows[:, None, :, None].expand(B, C, H, W),
        cols[:, None, None, :].expand(B, C, H, W),
    ]


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    net: Sequential,
    x_test: torch.Tensor,
    y_labels: torch.Tensor,
    batch_size: int = 256,
    n_bins: int = 15,
) -> dict[str, float]:
    """Return a dict of test metrics.

    Keys: ``test_acc``, ``mean_conf``, ``ece``, ``nll``, ``brier`` and the
    Remax output predictive-variance stats ``out_var_max`` / ``out_var_mean`` /
    ``out_var_p99``. NLL and Brier treat the Remax mean output ``mu`` as the
    predictive class-probability vector.
    """
    net.eval()
    correct = 0
    conf_chunks: list[torch.Tensor] = []
    hit_chunks: list[torch.Tensor] = []
    nll_chunks: list[torch.Tensor] = []
    brier_chunks: list[torch.Tensor] = []
    out_var_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            mu, var = net.forward(x_test[i : i + batch_size])
            yb = y_labels[i : i + batch_size]
            conf, pred = mu.max(dim=1)
            hit = pred == yb
            correct += hit.sum().item()
            conf_chunks.append(conf.detach())
            hit_chunks.append(hit.detach().to(conf.dtype))

            # NLL: -log p[true]. mu is the predictive probability vector; clamp
            # for numerical safety since the moment approximation can dip <=0.
            p = mu.clamp(min=1e-12)
            p_true = p.gather(1, yb.view(-1, 1)).squeeze(1)
            nll_chunks.append((-p_true.log()).detach())

            # Multi-class Brier: sum_k (p_k - onehot_k)^2, averaged over samples.
            onehot = torch.zeros_like(mu)
            onehot.scatter_(1, yb.view(-1, 1), 1.0)
            brier_chunks.append(((mu - onehot) ** 2).sum(dim=1).detach())

            out_var_chunks.append(var.detach().reshape(-1))

    conf_all = torch.cat(conf_chunks)
    hit_all = torch.cat(hit_chunks)
    ece = torch.zeros((), device=conf_all.device, dtype=conf_all.dtype)
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=conf_all.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf_all > lo) & (conf_all <= hi)
        if in_bin.any():
            weight = in_bin.float().mean()
            ece = ece + weight * (hit_all[in_bin].mean() - conf_all[in_bin].mean()).abs()

    out_var_all = torch.cat(out_var_chunks).float()
    net.train()
    return {
        "test_acc": correct / len(x_test),
        "mean_conf": conf_all.mean().item(),
        "ece": ece.item(),
        "nll": torch.cat(nll_chunks).mean().item(),
        "brier": torch.cat(brier_chunks).mean().item(),
        "out_var_max": out_var_all.max().item(),
        "out_var_mean": out_var_all.mean().item(),
        "out_var_p99": torch.quantile(out_var_all, 0.99).item(),
    }


def layer_variance_stats(
    net: Sequential,
    x_batch: torch.Tensor,
) -> tuple[list[dict], float, int]:
    """Manual eval-mode forward capturing each layer's output variance.

    Mirrors ``Sequential.forward`` but records ``Sa`` after every layer so we
    can watch how the activation variance propagates through depth. Returns
    ``(per_layer, global_max, global_max_idx)`` where ``per_layer`` is a list of
    ``{idx, name, var_max, var_mean}`` dicts, one per layer in ``net.layers``.
    """
    net.eval()
    ma = x_batch
    Sa = torch.zeros_like(x_batch)
    per_layer: list[dict] = []
    with torch.no_grad():
        for idx, layer in enumerate(net.layers):
            ma, Sa = layer.forward(ma, Sa)
            per_layer.append(
                {
                    "idx": idx,
                    "name": type(layer).__name__,
                    "var_max": Sa.max().item(),
                    "var_mean": Sa.mean().item(),
                }
            )
    net.train()
    gmax = max(per_layer, key=lambda d: d["var_max"])
    return per_layer, gmax["var_max"], gmax["idx"]


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def make_sigma_v_schedule(start, end, decay_epochs, shape, n_epochs):
    """Return f(epoch:int)->sigma_v.

    Anneals `start`->`end` over epochs 1..decay_epochs with the given shape,
    then holds at `end` for the remaining epochs. If `end` is None, sigma_v is
    fixed at `start` (the original fixed-noise behavior).
    """
    if end is None:
        return lambda epoch: start
    decay_epochs = min(decay_epochs, n_epochs)
    if decay_epochs <= 1:
        return lambda epoch: end

    def f(epoch):
        if epoch >= decay_epochs:
            return end
        t = (epoch - 1) / (decay_epochs - 1)   # 0.0 at ep1, 1.0 at decay_epochs
        if shape == "linear":
            return start + (end - start) * t
        if shape == "cosine":
            return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))
        if shape == "exp":     # geometric / log-linear (needs start,end > 0)
            return math.exp(math.log(start) + (math.log(end) - math.log(start)) * t)
        raise ValueError(f"unknown sigma_v schedule shape: {shape}")

    return f


def train(
    net: Sequential,
    x_train: torch.Tensor,
    y_train_oh: torch.Tensor,
    x_test: torch.Tensor,
    y_test_labels: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    sigma_v_fn,
    augment: bool,
    device: torch.device,
    run: RunDir,
    config: dict,
) -> float:
    """Training loop with optional GPU augmentation. Returns best test accuracy."""
    header = (
        f"\n  {'Epoch':>5}  {'σ_v':>6}  {'Acc':>7}  {'Conf':>6}  {'ECE':>6}  {'NLL':>6}"
        f"  {'Brier':>6}  {'oVarMax':>8}  {'lVarMax':>9}  {'@L':>3}  {'Time':>7}"
    )
    print(header)
    print("  " + "─" * (len(header) - 3))

    # Fixed test batch reused every epoch so per-layer variance is comparable.
    var_probe = x_test[: min(batch_size, len(x_test))]
    layer_var_log = run.path / "layer_variance.jsonl"

    best_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        sv = sigma_v_fn(epoch)
        perm = torch.randperm(x_train.size(0), device=device)
        x_s, y_s = x_train[perm], y_train_oh[perm]

        for i in range(0, len(x_s), batch_size):
            xb = x_s[i : i + batch_size]
            if augment:
                xb = gpu_augment(xb)
            net.step(xb, y_s[i : i + batch_size], sv)

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        m = evaluate(net, x_test, y_test_labels)
        per_layer, layer_var_max, layer_var_max_idx = layer_variance_stats(net, var_probe)
        acc = m["test_acc"]
        best_acc = max(best_acc, acc)
        print(
            f"  {epoch:5d}  {sv:6.4f}  {acc*100:6.2f}%  {m['mean_conf']:6.3f}  {m['ece']:6.3f}"
            f"  {m['nll']:6.3f}  {m['brier']:6.3f}  {m['out_var_max']:8.2e}"
            f"  {layer_var_max:9.2e}  {layer_var_max_idx:3d}  {wall:6.2f}s",
            flush=True,
        )
        run.append_metrics(
            epoch,
            test_acc=acc,
            mean_conf=m["mean_conf"],
            ece=m["ece"],
            nll=m["nll"],
            brier=m["brier"],
            out_var_max=m["out_var_max"],
            out_var_mean=m["out_var_mean"],
            out_var_p99=m["out_var_p99"],
            layer_var_max=layer_var_max,
            layer_var_max_idx=layer_var_max_idx,
            sigma_v=sv,
            wall_s=wall,
        )
        with open(layer_var_log, "a") as f:
            f.write(json.dumps({"epoch": epoch, "layers": per_layer}) + "\n")

        if epoch % config.get("checkpoint_interval", 10) == 0 or epoch == n_epochs:
            run.save_checkpoint(net, epoch, config)

    print("  " + "─" * 34)
    print(f"  Best test accuracy: {best_acc*100:.2f}%")
    return best_acc


# ---------------------------------------------------------------------------
#  Figure
# ---------------------------------------------------------------------------

def save_figure(run: RunDir) -> None:
    try:
        import csv
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping figure (pip install matplotlib)")
        return

    epochs, accs = [], []
    with open(run.metrics_csv) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            accs.append(float(row["test_acc"]) * 100)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, accs, color="C0", linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("CIFAR-10 ResNet-18 — TAGI test accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(run.figures / f"training_curve.{ext}", dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {run.figures}/")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(
    n_epochs: int = 100,
    batch_size: int = 128,
    sigma_v: float = 0.05,
    sigma_v_end: float | None = None,
    sigma_v_decay_epochs: int = 50,
    sigma_v_schedule: str = "linear",
    gain_w: float = 0.1,
    gain_b: float = 0.1,
    remax_approximation: str = "lognormal",
    remax_jacobian: str = "diag",
    remax_num_quad: int = 48,
    augment: bool = True,
    data_dir: str = "data",
    checkpoint_interval: int = 10,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """CIFAR-10 ResNet-18 benchmark.

    Args:
        sigma_v: Observation noise (fixed).
        remax_approximation: "lognormal" for cuTAGI parity or "laplace".
        remax_jacobian: "diag" or "full" for Laplace-Remax backward.
        augment: Apply random flip + crop augmentation each batch.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)

    print("=" * 60)
    print("  CIFAR-10 Classification — ResNet-18 — triton-tagi")
    print("  Stem+4 stages(64→128→256→512)+GAP → FC(512→10) → Remax")
    print(f"  Remax approximation: {remax_approximation} ({remax_jacobian})")
    print("=" * 60)
    if device == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    # ── Data ──
    print(f"\n  Loading CIFAR-10 from '{data_dir}'...", flush=True)
    x_train, y_train_oh, _, x_test, y_test_labels = load_cifar10(data_dir, dev)
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")
    print(f"  Input shape: {tuple(x_train.shape[1:])}")

    # ── Config ──
    config: dict = {
        "dataset": "cifar10",
        "arch": "resnet18",
        "optimizer": "tagi",
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "sigma_v": sigma_v,
        "sigma_v_start": sigma_v,
        "sigma_v_end": sigma_v_end,
        "sigma_v_decay_epochs": sigma_v_decay_epochs,
        "sigma_v_schedule": sigma_v_schedule,
        "gain_w": gain_w,
        "gain_b": gain_b,
        "remax_approximation": remax_approximation,
        "remax_jacobian": remax_jacobian,
        "remax_num_quad": remax_num_quad,
        "augment": augment,
        "checkpoint_interval": checkpoint_interval,
        "seed": seed,
        "device": device,
        "triton_tagi_version": "0.1.0",
    }

    # ── RunDir ──
    arch_tag = "resnet18" if remax_approximation == "lognormal" else f"resnet18_{remax_approximation}_{remax_jacobian}"
    run = RunDir("cifar10", arch_tag, "tagi")
    run.save_config(config)
    print(f"  Run directory: {run.path}")

    # ── Network ──
    kw = {"device": dev, "gain_w": gain_w, "gain_b": gain_b}

    net = Sequential(
        [
            # Stem: 32×32
            Conv2D(3, 64, 3, stride=1, padding=1, **kw),
            ReLU(),
            BatchNorm2D(64, **kw),
            # Stage 1: 32×32
            ResBlock(64, 64, stride=1, **kw),
            ResBlock(64, 64, stride=1, **kw),
            # Stage 2: 32→16
            ResBlock(64, 128, stride=2, **kw),
            ResBlock(128, 128, stride=1, **kw),
            # Stage 3: 16→8
            ResBlock(128, 256, stride=2, **kw),
            ResBlock(256, 256, stride=1, **kw),
            # Stage 4: 8→4
            ResBlock(256, 512, stride=2, **kw),
            ResBlock(512, 512, stride=1, **kw),
            # Head
            AvgPool2D(4),           # 4×4 → 1×1
            Flatten(),              # 512
            Linear(512, 10, **kw),
            Remax(
                approximation=remax_approximation,
                jacobian=remax_jacobian,
                num_quad=remax_num_quad,
            ),
        ],
        device=dev,
    )
    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")
    sigma_v_fn = make_sigma_v_schedule(
        sigma_v, sigma_v_end, sigma_v_decay_epochs, sigma_v_schedule, n_epochs
    )
    sv_desc = (
        f"{sigma_v}" if sigma_v_end is None
        else f"{sigma_v}→{sigma_v_end} by ep{min(sigma_v_decay_epochs, n_epochs)} ({sigma_v_schedule})"
    )
    print(
        f"\n  Epochs: {n_epochs}  |  Batch: {batch_size}  |  σ_v: {sv_desc}"
        f"  |  augment: {augment}  |  remax: {remax_approximation}/{remax_jacobian}"
    )

    # ── Train ──
    best_acc = train(
        net, x_train, y_train_oh, x_test, y_test_labels,
        n_epochs, batch_size, sigma_v_fn, augment, dev, run, config,
    )

    # ── Figure ──
    save_figure(run)
    print(f"\n  Results in: {run.path}")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CIFAR-10 ResNet-18 benchmark with TAGI"
    )
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sigma_v", type=float, default=0.05,
                        help="Observation noise (start value if --sigma_v_end is set).")
    parser.add_argument("--sigma_v_end", type=float, default=None,
                        help="If set, anneal sigma_v from --sigma_v to this value.")
    parser.add_argument("--sigma_v_decay_epochs", type=int, default=50,
                        help="Epoch by which sigma_v reaches --sigma_v_end, then held.")
    parser.add_argument("--sigma_v_schedule", choices=["linear", "cosine", "exp"],
                        default="linear", help="Annealing shape for sigma_v.")
    parser.add_argument("--gain_w", type=float, default=0.1)
    parser.add_argument("--gain_b", type=float, default=0.1)
    parser.add_argument(
        "--remax_approximation",
        choices=["lognormal", "laplace"],
        default="lognormal",
    )
    parser.add_argument(
        "--remax_jacobian",
        choices=["diag", "full"],
        default="diag",
    )
    parser.add_argument("--remax_num_quad", type=int, default=48)
    parser.add_argument("--no_augment", dest="augment", action="store_false",
                        help="Disable GPU augmentation")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.set_defaults(augment=True)
    args = parser.parse_args()
    main(**vars(args))
