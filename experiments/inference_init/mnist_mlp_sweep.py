"""MNIST MLP sweep for Inference-Based Initialization (IBI).

Reproduces the thesis Figure 4 acceptance targets in PLAN.md:
    - L=7, sigma_V=0.01, (sigma_M=1.0, sigma_Z=0.5) → ~96.8%  (He collapses)
    - L=5, sigma_V=0.05, (sigma_M=0.5, sigma_Z=0.5) → ~97.7%  (He collapses)

For each depth L in ``--depths``, the script sweeps a grid of (sigma_M, sigma_Z)
hyperparameters, runs ``inference_init`` once, then trains for ``--n_epochs``
epochs. A He-init baseline is run alongside each depth. Results are serialized
as JSON (one per depth) and two figures are written: a per-depth heatmap and a
combined panel showing all depths side-by-side.

Usage:
    # Single depth, 20 epochs, 3x3 grid (default)
    python experiments/inference_init/mnist_mlp_sweep.py --depths 5 --n_epochs 20

    # Full sweep across 5 depths, 4x4 grid, 20 epochs (long-running)
    python experiments/inference_init/mnist_mlp_sweep.py \\
        --depths 1 3 5 7 9 \\
        --sigma_m 0.25 0.5 1.0 2.0 --sigma_z 0.25 0.5 1.0 2.0 \\
        --n_epochs 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torchvision import datasets

from triton_tagi import Linear, ReLU, Sequential, inference_init

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RESULTS = HERE / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------


def load_mnist(data_dir: str, device: torch.device):
    train_ds = datasets.MNIST(data_dir, train=True, download=True)
    test_ds = datasets.MNIST(data_dir, train=False, download=True)

    x_train = train_ds.data.float().view(-1, 784) / 255.0
    x_test = test_ds.data.float().view(-1, 784) / 255.0
    mu, sigma = x_train.mean(), x_train.std()
    x_train = ((x_train - mu) / sigma).to(device)
    x_test = ((x_test - mu) / sigma).to(device)

    y_train_labels = train_ds.targets.to(device)
    y_test_labels = test_ds.targets.to(device)

    y_train_oh = torch.zeros(len(y_train_labels), 10, device=device)
    y_train_oh.scatter_(1, y_train_labels.unsqueeze(1), 1.0)
    return x_train, y_train_oh, x_test, y_test_labels


# ---------------------------------------------------------------------------
#  Build / train / eval
# ---------------------------------------------------------------------------


def build_mlp(depth: int, hidden: int, device: torch.device) -> Sequential:
    """Construct a depth-L MLP: 784 → [hidden] * depth → 10 with ReLU activations."""
    layers: list = []
    in_feat = 784
    for _ in range(depth):
        layers.append(Linear(in_feat, hidden, device=device))
        layers.append(ReLU())
        in_feat = hidden
    layers.append(Linear(in_feat, 10, device=device))
    return Sequential(layers, device=device)


def batch_iter(x: torch.Tensor, batch_size: int):
    """Yield fixed-size batches in the natural order (no shuffle)."""
    for i in range(0, len(x), batch_size):
        yield x[i : i + batch_size]


def evaluate(
    net: Sequential,
    x_test: torch.Tensor,
    y_labels: torch.Tensor,
    batch_size: int = 1024,
) -> float:
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            mu, _ = net.forward(x_test[i : i + batch_size])
            correct += (mu.argmax(dim=1) == y_labels[i : i + batch_size]).sum().item()
    net.train()
    return correct / len(x_test)


def train_run(
    depth: int,
    hidden: int,
    sigma_v: float,
    ibi: bool,
    sigma_m: float | None,
    sigma_z: float | None,
    x_train: torch.Tensor,
    y_train_oh: torch.Tensor,
    x_test: torch.Tensor,
    y_test_labels: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict:
    """Single training run. Returns {best_acc, final_acc, diverged, per_epoch_acc}."""
    torch.manual_seed(seed)
    net = build_mlp(depth, hidden, device)

    if ibi:
        assert sigma_m is not None and sigma_z is not None
        calib_loader = list(batch_iter(x_train, batch_size))
        inference_init(net, calib_loader, sigma_m, sigma_z)

    per_epoch = []
    best_acc = 0.0
    diverged = False
    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(x_train.size(0), device=device)
        x_s, y_s = x_train[perm], y_train_oh[perm]
        for i in range(0, len(x_s), batch_size):
            net.step(x_s[i : i + batch_size], y_s[i : i + batch_size], sigma_v)

        acc = evaluate(net, x_test, y_test_labels)
        per_epoch.append(acc)
        best_acc = max(best_acc, acc)
        if not torch.isfinite(net.forward(x_test[:1])[0]).all():
            diverged = True
            break
    return {
        "best_acc": best_acc,
        "final_acc": per_epoch[-1] if per_epoch else 0.0,
        "per_epoch_acc": per_epoch,
        "diverged": diverged,
    }


# ---------------------------------------------------------------------------
#  Heatmap
# ---------------------------------------------------------------------------


def save_heatmap(
    accs: list[list[float]],
    sigma_m_vals: list[float],
    sigma_z_vals: list[float],
    title: str,
    out_path: Path,
    he_baseline: float | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not installed, skipping heatmap")
        return

    arr = np.array(accs) * 100.0
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(arr, cmap="viridis", origin="lower", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(sigma_z_vals)))
    ax.set_xticklabels([f"{z:g}" for z in sigma_z_vals])
    ax.set_yticks(range(len(sigma_m_vals)))
    ax.set_yticklabels([f"{m:g}" for m in sigma_m_vals])
    ax.set_xlabel(r"$\sigma_Z$")
    ax.set_ylabel(r"$\sigma_M$")
    subtitle = title
    if he_baseline is not None:
        subtitle += f"  (He baseline: {he_baseline*100:.2f}%)"
    ax.set_title(subtitle)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(
                j, i, f"{arr[i, j]:.1f}", ha="center", va="center",
                color="white" if arr[i, j] < 50 else "black", fontsize=9,
            )
    fig.colorbar(im, ax=ax, label="Test acc (%)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}", dpi=150)
    plt.close(fig)


def save_combined_panel(
    depth_results: list[dict],
    sigma_m_vals: list[float],
    sigma_z_vals: list[float],
    sigma_v: float,
    n_epochs: int,
    out_path: Path,
) -> None:
    """Combined panel: one heatmap per depth, arranged horizontally."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    n = len(depth_results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)
    axes = axes[0]
    im = None
    for ax, r in zip(axes, depth_results, strict=True):
        arr = np.array(r["grid_best_acc"]) * 100.0
        im = ax.imshow(arr, cmap="viridis", origin="lower", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(sigma_z_vals)))
        ax.set_xticklabels([f"{z:g}" for z in sigma_z_vals])
        ax.set_yticks(range(len(sigma_m_vals)))
        ax.set_yticklabels([f"{m:g}" for m in sigma_m_vals])
        ax.set_xlabel(r"$\sigma_Z$")
        ax.set_ylabel(r"$\sigma_M$")
        he = r["he_baseline"]["best_acc"] * 100.0
        ax.set_title(f"L={r['depth']}  IBI  (He: {he:.1f}%)")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(
                    j, i, f"{arr[i, j]:.1f}", ha="center", va="center",
                    color="white" if arr[i, j] < 50 else "black", fontsize=8,
                )
    fig.suptitle(
        f"MNIST MLP IBI sweep  σ_V={sigma_v}  epochs={n_epochs}  (best test acc %)"
    )
    fig.colorbar(im, ax=axes.tolist(), shrink=0.85, label="Test acc (%)")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_path}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Sweep
# ---------------------------------------------------------------------------


def sweep(
    depth: int,
    hidden: int,
    sigma_v: float,
    sigma_m_vals: list[float],
    sigma_z_vals: list[float],
    n_epochs: int,
    batch_size: int,
    seed: int,
    data: tuple,
    device: torch.device,
    stamp: str,
) -> dict:
    """Run He baseline + (σ_M, σ_Z) grid for one depth. Return full result dict."""
    x_train, y_train_oh, x_test, y_test_labels = data
    print("=" * 64)
    print(f"  IBI sweep: depth={depth}, sigma_V={sigma_v}, epochs={n_epochs}")
    print(f"  sigma_M: {sigma_m_vals}   sigma_Z: {sigma_z_vals}")
    print("=" * 64)

    # ── He baseline ──
    t0 = time.perf_counter()
    he = train_run(
        depth=depth, hidden=hidden, sigma_v=sigma_v,
        ibi=False, sigma_m=None, sigma_z=None,
        x_train=x_train, y_train_oh=y_train_oh,
        x_test=x_test, y_test_labels=y_test_labels,
        n_epochs=n_epochs, batch_size=batch_size, seed=seed, device=device,
    )
    print(f"  He baseline: best={he['best_acc']*100:6.2f}%  "
          f"final={he['final_acc']*100:6.2f}%  "
          f"diverged={he['diverged']}  ({time.perf_counter()-t0:.1f}s)",
          flush=True)

    # ── IBI grid ──
    grid_best = [[0.0 for _ in sigma_z_vals] for _ in sigma_m_vals]
    grid_final = [[0.0 for _ in sigma_z_vals] for _ in sigma_m_vals]
    grid_diverged = [[False for _ in sigma_z_vals] for _ in sigma_m_vals]
    runs = []
    for i, sm in enumerate(sigma_m_vals):
        for j, sz in enumerate(sigma_z_vals):
            t0 = time.perf_counter()
            r = train_run(
                depth=depth, hidden=hidden, sigma_v=sigma_v,
                ibi=True, sigma_m=sm, sigma_z=sz,
                x_train=x_train, y_train_oh=y_train_oh,
                x_test=x_test, y_test_labels=y_test_labels,
                n_epochs=n_epochs, batch_size=batch_size, seed=seed, device=device,
            )
            grid_best[i][j] = r["best_acc"]
            grid_final[i][j] = r["final_acc"]
            grid_diverged[i][j] = r["diverged"]
            runs.append({"sigma_m": sm, "sigma_z": sz, **r})
            print(f"  IBI σM={sm:.2f} σZ={sz:.2f}: "
                  f"best={r['best_acc']*100:6.2f}%  "
                  f"final={r['final_acc']*100:6.2f}%  "
                  f"diverged={r['diverged']}  "
                  f"({time.perf_counter()-t0:.1f}s)",
                  flush=True)

    # ── Save per-depth results ──
    tag = f"L{depth}_sv{sigma_v:g}_{stamp}"
    result = {
        "depth": depth,
        "hidden": hidden,
        "sigma_v": sigma_v,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "seed": seed,
        "sigma_m_vals": sigma_m_vals,
        "sigma_z_vals": sigma_z_vals,
        "he_baseline": he,
        "ibi_runs": runs,
        "grid_best_acc": grid_best,
        "grid_final_acc": grid_final,
        "grid_diverged": grid_diverged,
    }
    with open(RESULTS / f"{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    save_heatmap(
        grid_best, sigma_m_vals, sigma_z_vals,
        title=f"MNIST MLP  L={depth}  σ_V={sigma_v}  best test acc",
        out_path=FIGURES / f"heatmap_{tag}",
        he_baseline=he["best_acc"],
    )
    print(f"  -> {RESULTS / f'{tag}.json'}")
    return result


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="MNIST MLP IBI sweep")
    p.add_argument("--depths", type=int, nargs="+", default=[5],
                   help="One or more hidden-layer depths (L). Default: [5].")
    p.add_argument("--hidden", type=int, default=256, help="Hidden layer width")
    p.add_argument("--sigma_v", type=float, default=0.05)
    p.add_argument("--sigma_m", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--sigma_z", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    data = load_mnist(args.data_dir, device)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    t_total = time.perf_counter()
    depth_results = []
    for depth in args.depths:
        r = sweep(
            depth=depth,
            hidden=args.hidden,
            sigma_v=args.sigma_v,
            sigma_m_vals=args.sigma_m,
            sigma_z_vals=args.sigma_z,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            data=data,
            device=device,
            stamp=stamp,
        )
        depth_results.append(r)

    if len(depth_results) > 1:
        panel_tag = (
            f"panel_L{'-'.join(str(d) for d in args.depths)}_sv{args.sigma_v:g}_{stamp}"
        )
        save_combined_panel(
            depth_results,
            args.sigma_m,
            args.sigma_z,
            args.sigma_v,
            args.n_epochs,
            FIGURES / panel_tag,
        )
        print(f"\n  Combined panel: {FIGURES / f'{panel_tag}.png'}")

    print(f"\n  Total wall time: {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
