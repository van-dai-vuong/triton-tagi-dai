"""
Heteroscedastic 1-D Regression — triton-tagi example.

Demonstrates TAGI-V: the network learns both the mean prediction and the
input-dependent noise variance simultaneously.  The output layer has 2 neurons:
  - index 0 (even): mean prediction
  - index 1 (odd):  noise variance prediction (passed through EvenExp)

This mirrors cuTAGI's heteroscedastic regression exactly: the variance head is
the exponential activation (`SplitActivation(Exp())` in cuTAGI), and the output
update is the AGVI heteros kernel — a 1:1 port of cuTAGI's
`update_delta_z_cuda_heteros`. Results match cuTAGI within fp32 tolerance.

Usage:
    python examples/regression_heteros.py
    python examples/regression_heteros.py --n_epochs 100 --sigma_v 1.0
    python examples/regression_heteros.py --data_dir /path/to/toy_example
    python examples/regression_heteros.py --help
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from triton_tagi import EvenExp, Linear, ReLU, Sequential
from triton_tagi.checkpoint import RunDir


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

def generate_data(
    n_train: int = 800,
    n_test: int = 500,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic heteroscedastic 1-D data: y = sin(x) + ε(x).

    Noise standard deviation grows linearly: σ(x) = 0.05 + 0.3 · |x|,
    so uncertainty bands widen visibly away from x = 0.

    Args:
        n_train: Number of training samples.
        n_test:  Number of test samples.
        seed:    Random seed.

    Returns:
        x_train, y_train, x_test, y_test as float32 arrays of shape (N, 1).
    """
    rng = np.random.default_rng(seed)
    x_tr = rng.uniform(-4.0, 4.0, n_train).astype(np.float32)
    x_te = np.linspace(-4.0, 4.0, n_test, dtype=np.float32)

    def _sample(x: np.ndarray) -> np.ndarray:
        noise_std = 0.05 + 0.3 * np.abs(x)
        return np.sin(x) + rng.normal(0.0, noise_std).astype(np.float32)

    y_tr = _sample(x_tr)
    y_te = _sample(x_te)
    return x_tr.reshape(-1, 1), y_tr.reshape(-1, 1), x_te.reshape(-1, 1), y_te.reshape(-1, 1)


def load_csv_data(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cuTAGI-format heteroscedastic noise CSVs.

    Expects: x_train_noise.csv, y_train_noise.csv, x_test_noise.csv, y_test_noise.csv.

    Args:
        data_dir: Directory containing the four CSV files.

    Returns:
        x_train, y_train, x_test, y_test as float32 arrays of shape (N, 1).
    """
    def _read(path: Path) -> np.ndarray:
        return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32).reshape(-1, 1)

    d = Path(data_dir)
    return (
        _read(d / "x_train_noise.csv"),
        _read(d / "y_train_noise.csv"),
        _read(d / "x_test_noise.csv"),
        _read(d / "y_test_noise.csv"),
    )


def normalise(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple]:
    """Z-score standardisation using training statistics."""
    x_mean, x_std = x_tr.mean(), x_tr.std() + 1e-8
    y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-8
    return (
        (x_tr - x_mean) / x_std,
        (y_tr - y_mean) / y_std,
        (x_te - x_mean) / x_std,
        (y_te - y_mean) / y_std,
        (float(x_mean), float(x_std), float(y_mean), float(y_std)),
    )


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def log_likelihood(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    pred_std: np.ndarray,
) -> float:
    """Mean Gaussian log-likelihood."""
    ll = -0.5 * np.log(2 * math.pi * pred_std**2) - 0.5 * ((y_true - y_pred) / pred_std) ** 2
    return float(ll.mean())


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train(
    net: Sequential,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    n_epochs: int,
    batch_size: int,
    sigma_v: float,
    device: torch.device,
    run: RunDir,
    config: dict,
) -> None:
    """Train one epoch at a time, logging MSE on even-indexed outputs."""
    rng = np.random.default_rng(42)
    n = len(x_tr)
    print(f"\n  {'Epoch':>5}  {'Train MSE':>10}  {'Time':>7}")
    print("  " + "─" * 28)

    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        perm = rng.permutation(n)
        x_s, y_s = x_tr[perm], y_tr[perm]

        sq_errs = []
        for i in range(0, n, batch_size):
            xb = torch.tensor(x_s[i : i + batch_size], device=device)
            yb = torch.tensor(y_s[i : i + batch_size], device=device)
            # net.step auto-selects heteros kernel: output (B,2) vs target (B,1)
            mu_pred, _ = net.step(xb, yb, sigma_v)
            # even columns = mean predictions
            mu_mean = mu_pred[:, 0:1].cpu().numpy()
            sq_errs.append(((mu_mean - y_s[i : i + batch_size]) ** 2).mean())

        train_mse = float(np.mean(sq_errs))
        wall = time.perf_counter() - t0
        print(f"  {epoch:5d}  {train_mse:10.4f}  {wall:6.3f}s")
        run.append_metrics(epoch, train_mse=train_mse, wall_s=wall)

        if epoch % config.get("checkpoint_interval", 10) == 0 or epoch == n_epochs:
            run.save_checkpoint(net, epoch, config)


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    net: Sequential,
    x_te: np.ndarray,
    sigma_v: float,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted mean and total predictive std in normalised space."""
    net.eval()
    mu_list, total_std_list = [], []

    for i in range(0, len(x_te), batch_size):
        xb = torch.tensor(x_te[i : i + batch_size], device=device)
        with torch.no_grad():
            mu_2k, S_2k = net.forward(xb)

        # Even: mean prediction; Odd: noise variance (aleatoric, after EvenExp)
        mu_pred = mu_2k[:, 0:1].cpu().numpy()          # (B, 1)
        noise_var = mu_2k[:, 1:2].cpu().numpy()        # aleatoric σ²_v (predicted)
        epist_var = S_2k[:, 0:1].cpu().numpy()         # epistemic σ²_ep

        total_std = np.sqrt(np.maximum(epist_var + noise_var, 1e-8))
        mu_list.append(mu_pred)
        total_std_list.append(total_std)

    net.train()
    return np.concatenate(mu_list), np.concatenate(total_std_list)


# ---------------------------------------------------------------------------
#  Figure
# ---------------------------------------------------------------------------

def save_figure(
    x_tr_orig: np.ndarray,
    y_tr_orig: np.ndarray,
    x_te_orig: np.ndarray,
    mu_pred_orig: np.ndarray,
    std_pred_orig: np.ndarray,
    run: RunDir,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping figure (pip install matplotlib)")
        return

    sort_idx = np.argsort(x_te_orig.ravel())
    x_p = x_te_orig.ravel()[sort_idx]
    mu_p = mu_pred_orig.ravel()[sort_idx]
    std_p = std_pred_orig.ravel()[sort_idx]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(x_p, mu_p - 3 * std_p, mu_p + 3 * std_p, alpha=0.12, color="C0", label="±3σ")
    ax.fill_between(x_p, mu_p - std_p, mu_p + std_p, alpha=0.28, color="C0", label="±1σ")
    ax.plot(x_p, mu_p, color="C0", linewidth=1.5, label="Prediction mean")
    ax.scatter(
        x_tr_orig.ravel(),
        y_tr_orig.ravel(),
        s=6,
        alpha=0.35,
        color="k",
        zorder=4,
        label="Training data",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("TAGI-V Heteroscedastic Regression — learned noise variance")
    ax.legend(fontsize=8)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(run.figures / f"predictions.{ext}", dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {run.figures}/")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(
    n_epochs: int = 50,
    batch_size: int = 10,
    sigma_v: float = 1.0,
    hidden: int = 128,
    gain_w: float = 1.0,
    gain_b: float = 1.0,
    data_dir: str | None = None,
    checkpoint_interval: int = 10,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Heteroscedastic TAGI regression demo.

    Note: sigma_v is passed to net.step() but ignored by the heteros update
    kernel — the observation noise is predicted by the network itself.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device)
    print("=" * 60)
    print("  TAGI-V Heteroscedastic Regression — triton-tagi")
    print("=" * 60)
    if device == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    # ── Data ──
    if data_dir is not None:
        print(f"\n  Loading data from {data_dir}")
        x_tr_raw, y_tr_raw, x_te_raw, y_te_raw = load_csv_data(data_dir)
    else:
        print("\n  Generating synthetic data (y = sin(x) + ε(x), σ(x)=0.05+0.3|x|)")
        x_tr_raw, y_tr_raw, x_te_raw, y_te_raw = generate_data(seed=seed)

    x_tr, y_tr, x_te, y_te, stats = normalise(x_tr_raw, y_tr_raw, x_te_raw, y_te_raw)
    x_mean, x_std, y_mean, y_std = stats
    print(f"  Train: {len(x_tr)}  |  Test: {len(x_te)}")
    print("  Note: EvenExp variance head + AGVI heteros update — matches cuTAGI (fp32).")

    # ── Config ──
    config: dict = {
        "dataset": "regression_heteros_1d",
        "arch": f"mlp_1-{hidden}-{hidden}-2",
        "optimizer": "tagi",
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "sigma_v": sigma_v,
        "hidden": hidden,
        "gain_w": gain_w,
        "gain_b": gain_b,
        "checkpoint_interval": checkpoint_interval,
        "seed": seed,
        "device": device,
        "triton_tagi_version": "0.1.0",
        "note": "EvenExp variance head + AGVI heteros update (matches cuTAGI)",
    }

    # ── RunDir ──
    run = RunDir("regression_heteros_1d", f"mlp_1-{hidden}-{hidden}-2", "tagi")
    run.save_config(config)
    print(f"\n  Run directory: {run.path}")

    # ── Network ──
    # Output has 2 neurons: [mean_prediction, noise_variance_prediction]
    # EvenExp applies exp to index 1 (noise variance) → always positive.
    # Mirrors cuTAGI's `SplitActivation(Exp())` (examples/regression_heteros.py).
    net = Sequential(
        [
            Linear(1, hidden, device=dev, gain_w=gain_w, gain_b=gain_b),
            ReLU(),
            Linear(hidden, hidden, device=dev, gain_w=gain_w, gain_b=gain_b),
            ReLU(),
            Linear(hidden, 2, device=dev, gain_w=gain_w, gain_b=gain_b),
            EvenExp(half_width=1),
        ],
        device=dev,
    )
    print(f"\n{net}")
    print(f"  Parameters: {net.num_parameters():,}")
    print(f"\n  Epochs: {n_epochs}  |  Batch: {batch_size}")

    # ── Train ──
    train(net, x_tr, y_tr, n_epochs, batch_size, sigma_v, dev, run, config)

    # ── Evaluate ──
    mu_norm, std_norm = evaluate(net, x_te, sigma_v, dev, batch_size)

    # Unstandardise (only mean and std scale with y_std)
    mu_orig = mu_norm * y_std + y_mean
    std_orig = std_norm * y_std

    test_mse = mse(mu_orig, y_te_raw.ravel())
    test_ll = log_likelihood(mu_orig, y_te_raw.ravel(), std_orig)

    print("\n  " + "─" * 44)
    print(f"  Test MSE           : {test_mse:.4f}")
    print(f"  Test log-likelihood: {test_ll:.4f}")
    print("  " + "─" * 44)

    # ── Figure ──
    save_figure(x_tr_raw, y_tr_raw, x_te_raw, mu_orig, std_orig, run)
    print(f"\n  Results in: {run.path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TAGI-V heteroscedastic regression: network learns noise variance"
    )
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--sigma_v",
        type=float,
        default=1.0,
        help="Passed to net.step() but unused by the heteros update kernel.",
    )
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--gain_w", type=float, default=1.0)
    parser.add_argument("--gain_b", type=float, default=1.0)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to cuTAGI toy_example/ directory (uses *_noise.csv files).",
    )
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    main(**vars(args))
