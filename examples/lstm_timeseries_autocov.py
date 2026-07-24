"""
Multi-step-ahead time-series forecasting with a Bayesian LSTM (autocov).

Trains an autocov LSTM one-step-ahead on a **noisy sine wave**, then forecasts
many steps into the future **autoregressively**: each prediction (with its
predictive variance) is fed back in as the next input, so the forecast
uncertainty compounds with the horizon — the natural Bayesian behavior.

    training  : window x_{t-T..t-1}  ──LSTM──▶  x̂_t          (teacher-forced)
    forecast  : x̂_t fed back as input for x̂_{t+1}, x̂_{t+2}, …  (recursive)

No optimizer / BPTT code: unrolling the LSTM in ``forward`` builds the graph and
``observe()`` runs backprop-through-time and updates every parameter in place.

Usage:
    python examples/lstm_timeseries_autocov.py
    python examples/lstm_timeseries_autocov.py --n_epochs 80 --hidden 32 --window 20
    python examples/lstm_timeseries_autocov.py --noise_std 0.1 --horizon 100
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from triton_tagi.autocov import Linear, Module, tensor
from triton_tagi.lstm import LSTM


# ---------------------------------------------------------------------------
#  Model:  LSTM  →  Linear readout
# ---------------------------------------------------------------------------

class LSTMRegressor(Module):
    def __init__(self, hidden: int, rng=None, device=None,
                 gain_w: float = 0.5, gain_b: float = 0.5):
        super().__init__()
        self.lstm = LSTM(1, hidden, rng=rng, device=device, gain_w=gain_w, gain_b=gain_b)
        self.readout = Linear(hidden, 1, rng=rng, device=device, gain_w=gain_w, gain_b=gain_b)

    def forward(self, seq):
        # seq: list of T tensors, each (B, 1).
        h = self.lstm(seq, return_sequence=False)   # layer 1: T hidden states, each (B, hidden)
        return self.readout(h)                       # predict from the top-layer last hidden


# ---------------------------------------------------------------------------
#  Data — a noisy sine wave + sliding-window (x_{t-T..t-1} -> x_t) set
# ---------------------------------------------------------------------------

def make_series(n: int = 300, periods: float = 6.0, noise_std: float = 0.05,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (noisy, clean) sine series of length n."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, periods * 2 * math.pi, n)
    clean = np.sin(t).astype(np.float32)
    noisy = (clean + rng.normal(0.0, noise_std, size=n)).astype(np.float32)
    return noisy, clean


def windowize(series: np.ndarray, window: int):
    """Return X (n_win, window) and Y (n_win,): each row predicts the next value."""
    X, Y = [], []
    for i in range(len(series) - window):
        X.append(series[i : i + window])
        Y.append(series[i + window])
    return np.stack(X).astype(np.float32), np.array(Y, dtype=np.float32).reshape(-1, 1)


def to_seq(X: torch.Tensor, var=0.0):
    """(B, T) window batch -> list of T tensors, each (B, 1). ``var`` may be a
    scalar or a length-T sequence of per-timestep variances (deterministic
    history has var=0; fed-back predictions carry their predictive variance)."""
    T = X.shape[1]
    vs = [var] * T if np.isscalar(var) else var
    return [tensor(X[:, t : t + 1], var=float(vs[t])) for t in range(T)]


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


# ---------------------------------------------------------------------------
#  Autoregressive multi-step-ahead forecast (uncertainty compounds)
# ---------------------------------------------------------------------------

def forecast(net, seed_window: np.ndarray, horizon: int, sigma_v: float,
             device: torch.device):
    """Roll the model forward ``horizon`` steps, feeding each prediction (and its
    predictive variance) back in as the next input. Returns (mu, std) arrays in
    the model's (normalised) space, both shape (horizon,)."""
    net.eval()
    win_mu = list(map(float, seed_window))     # observed history (deterministic)
    win_var = [0.0] * len(seed_window)
    mu_out, std_out = [], []
    for _ in range(horizon):
        xb = torch.tensor(win_mu, device=device).reshape(1, -1)   # (1, T)
        out = net(to_seq(xb, var=win_var))
        mu = float(out.mu.reshape(-1)[0])
        # full predictive variance of the predicted observation
        pvar = float(out.var.reshape(-1)[0]) + sigma_v ** 2
        mu_out.append(mu)
        std_out.append(math.sqrt(max(pvar, 0.0)))
        # slide the window: drop oldest, append the (uncertain) prediction
        win_mu = win_mu[1:] + [mu]
        win_var = win_var[1:] + [pvar]
    net.train()
    return np.array(mu_out), np.array(std_out)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(
    n_epochs: int = 80,
    hidden: int = 32,
    window: int = 1,
    noise_std: float = 0.05,
    sigma_v: float = 0.1,
    horizon: int | None = None,
    gain_w: float = 0.5,
    gain_b: float = 0.5,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    torch.manual_seed(seed)
    dev = torch.device(device)
    print("=" * 60)
    print("  Multi-step forecasting — Bayesian LSTM (autocov)")
    print("=" * 60)

    # ── Data: noisy sine ──
    noisy, clean = make_series(noise_std=noise_std, seed=seed)
    X, Y = windowize(noisy, window)
    n_train = int(0.7 * len(X))
    mu, sd = noisy[: n_train + window].mean(), noisy[: n_train + window].std() + 1e-8
    Xn, Yn = (X - mu) / sd, (Y - mu) / sd

    x_tr = torch.tensor(Xn[:n_train], device=dev)
    y_tr = torch.tensor(Yn[:n_train], device=dev)
    n_test = len(X) - n_train
    if horizon is None:
        horizon = n_test
    horizon = min(horizon, n_test)
    print(f"  Noisy sine: {len(noisy)} pts (noise σ={noise_std})  |  window T={window}")
    print(f"  Train windows: {n_train}  |  forecast horizon: {horizon} steps")

    # ── Model ──
    rng = torch.Generator(device=dev).manual_seed(seed)
    net = LSTMRegressor(hidden, rng=rng, device=device, gain_w=gain_w, gain_b=gain_b)
    net.assign_names()
    print(f"  LSTM(1→{hidden}) → LSTM({hidden}→{hidden}) → Linear({hidden}→1)"
          f"  |  parameters: {net.num_parameters:,}")

    # ── Train (one-step-ahead, teacher-forced) ──
    print(f"\n  {'Epoch':>5}  {'Train MSE':>10}  {'Time':>7}")
    print("  " + "─" * 28)
    var_v = sigma_v ** 2
    seq_tr = to_seq(x_tr, var=0.0)
    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        out = net(seq_tr)                        # forward builds the BPTT graph
        train_mse = mse(out.mu.detach().cpu().numpy(), Yn[:n_train])
        out.observe(y_tr, var_v=var_v)           # automatic BPTT + capped update
        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            print(f"  {epoch:5d}  {train_mse:10.4f}  {time.perf_counter()-t0:6.2f}s", flush=True)

    # ── Multi-step-ahead forecast (autoregressive, from train/test boundary) ──
    seed_window = Xn[n_train]                    # last observed window before the test region
    f_mu, f_std = forecast(net, seed_window, horizon, sigma_v, dev)

    # Un-normalise
    pred = f_mu * sd + mu
    pred_std = f_std * sd
    truth_noisy = Y[n_train : n_train + horizon].ravel()
    truth_clean = clean[n_train + window : n_train + window + horizon]

    fc_mse = mse(pred, truth_clean)
    print("  " + "─" * 28)
    print(f"  Multi-step forecast ({horizon} steps, recursive):")
    print(f"    RMSE vs clean signal : {math.sqrt(fc_mse):.4f}  (amplitude 1.0)")
    print(f"    ±1σ band grows: step 1 = {pred_std[0]:.3f}  →  step {horizon} = {pred_std[-1]:.3f}")

    # ── Figure ──
    try:
        import matplotlib.pyplot as plt
        idx = np.arange(horizon)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(idx, truth_clean, "k-", lw=1.2, label="true (clean) sine")
        ax.plot(idx, truth_noisy, "k.", ms=3, alpha=0.5, label="true (noisy)")
        ax.plot(idx, pred, "C0-", lw=1.5, label="LSTM forecast")
        ax.fill_between(idx, pred - 3 * pred_std, pred + 3 * pred_std,
                        color="C0", alpha=0.2, label="±3σ")
        ax.set_title(f"Bayesian LSTM — {horizon}-step recursive forecast (noisy sine)")
        ax.set_xlabel("forecast step"); ax.set_ylabel("value"); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("lstm_forecast.png", dpi=150)
        plt.close(fig)
        print("  Figure saved to lstm_forecast.png")
    except ImportError:
        print("  (matplotlib not installed — skipping figure)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian LSTM multi-step forecasting (autocov)")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=20)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--sigma_v", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=None,
                        help="Forecast steps (default: full test region)")
    parser.add_argument("--gain_w", type=float, default=0.5)
    parser.add_argument("--gain_b", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(**vars(args))
