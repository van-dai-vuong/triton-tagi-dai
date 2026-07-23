"""
CIFAR-10 ResNet-18 — autocov (autograd-for-TAGI) example.

Same architecture, data, and training recipe as ``examples/cifar10_resnet18.py``,
but the network is defined with the :mod:`triton_tagi.autocov` graph engine
instead of ``Sequential``. A Torch-style ``ResBlock`` module is built from
autocov ops (``Conv2D``, ``BatchNorm2D``, ``relu``, and the residual ``add``),
and ``ResNet18`` is assembled from those blocks. Training uses no explicit
``step``: build the graph in ``forward`` and call ``out.observe(y, var_v)`` —
the reverse sweep updates every parameter in place.

Architecture (CIFAR-10 adaptation — 3×3 stem, no max-pool):
    Stem:    Conv(3→64, 3×3, p=1) → ReLU → BN           [32×32]
    Stage 1: ResBlock(64,  64,  s=1) × 2                 [32×32]
    Stage 2: ResBlock(64,  128, s=2) + ResBlock(128, 128) [16×16]
    Stage 3: ResBlock(128, 256, s=2) + ResBlock(256, 256) [8×8]
    Stage 4: ResBlock(256, 512, s=2) + ResBlock(512, 512) [4×4]
    Head:    AvgPool(4) → Flatten → FC(512→10) → Remax

Each ResBlock: Conv(3×3,s) → ReLU → BN → Conv(3×3) → ReLU → BN + shortcut.
Projection shortcut (stride>1 or ch mismatch): Conv(2×2, s=2) → ReLU → BN.

Usage:
    python examples/cifar10_resnet18_autocov.py
    python examples/cifar10_resnet18_autocov.py --n_epochs 30
    python examples/cifar10_resnet18_autocov.py --data_dir /path/to/data --no_augment
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from triton_tagi.autocov import (
    AvgPool2D,
    BatchNorm2D,
    Conv2D,
    Flatten,
    Linear,
    Module,
    add,
    relu,
    remax,
    tensor,
)
from triton_tagi.checkpoint import RunDir


# ---------------------------------------------------------------------------
#  Network — autocov ResBlock and ResNet-18
# ---------------------------------------------------------------------------

class ResBlock(Module):
    """TAGI residual block built from autocov ops (matches triton_tagi.ResBlock).

    Main path:  Conv(3×3,s) → ReLU → BN → Conv(3×3) → ReLU → BN
    Shortcut:   identity, or (stride>1 / channel change) Conv(2×2,s) → ReLU → BN
    Merge:      out = main + shortcut     (autocov ``add``; no post-activation)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        gain_w: float = 0.1,
        gain_b: float = 0.1,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        kw = dict(gain_w=gain_w, gain_b=gain_b, device=device)

        # ── Main path ── (stride>1 uses cuTAGI's right-bottom padding_type=2)
        self.conv1 = Conv2D(
            in_ch, out_ch, 3, stride=stride, padding=1,
            padding_type=2 if stride > 1 else 1, **kw,
        )
        self.bn1 = BatchNorm2D(out_ch, preserve_var=False, **kw)
        self.conv2 = Conv2D(out_ch, out_ch, 3, stride=1, padding=1, **kw)
        self.bn2 = BatchNorm2D(out_ch, preserve_var=False, **kw)

        # ── Shortcut path ──
        self.use_proj = (stride != 1) or (in_ch != out_ch)
        if self.use_proj:
            self.proj_conv = Conv2D(in_ch, out_ch, 2, stride=stride, padding=0, **kw)
            self.proj_bn = BatchNorm2D(out_ch, preserve_var=False, **kw)

    def forward(self, x):
        # Main path
        z = self.bn1(relu(self.conv1(x)))
        z = self.bn2(relu(self.conv2(z)))
        # Shortcut path
        s = self.proj_bn(relu(self.proj_conv(x))) if self.use_proj else x
        # Merge (residual add; variances add under the diagonal approximation)
        return add(z, s)


class ResNet18(Module):
    """CIFAR-10 ResNet-18 assembled from autocov :class:`ResBlock`s."""

    def __init__(
        self,
        num_classes: int = 10,
        gain_w: float = 0.1,
        gain_b: float = 0.1,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        kw = dict(gain_w=gain_w, gain_b=gain_b, device=device)

        # Stem: 32×32
        self.stem_conv = Conv2D(3, 64, 3, stride=1, padding=1, **kw)
        self.stem_bn = BatchNorm2D(64, **kw)

        # 4 stages × 2 blocks
        self.b1a = ResBlock(64, 64, 1, **kw)
        self.b1b = ResBlock(64, 64, 1, **kw)
        self.b2a = ResBlock(64, 128, 2, **kw)
        self.b2b = ResBlock(128, 128, 1, **kw)
        self.b3a = ResBlock(128, 256, 2, **kw)
        self.b3b = ResBlock(256, 256, 1, **kw)
        self.b4a = ResBlock(256, 512, 2, **kw)
        self.b4b = ResBlock(512, 512, 1, **kw)
        self._blocks = [
            self.b1a, self.b1b, self.b2a, self.b2b,
            self.b3a, self.b3b, self.b4a, self.b4b,
        ]

        # Head
        self.pool = AvgPool2D(4)   # 4×4 → 1×1
        self.flat = Flatten()      # 512
        self.fc = Linear(512, num_classes, **kw)

    def forward(self, x):
        h = self.stem_bn(relu(self.stem_conv(x)))
        for block in self._blocks:
            h = block(h)
        h = self.fc(self.flat(self.pool(h)))
        return remax(h)   # Remax classification head (lognormal, cuTAGI parity)


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(data_dir="data", device=torch.device("cpu")):
    """Load CIFAR-10 as (N, 3, 32, 32) tensors on ``device``."""
    norm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD)]
    )
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=norm)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=norm)

    x_train = torch.stack([img for img, _ in train_ds]).to(device)
    y_train = torch.tensor([lbl for _, lbl in train_ds], device=device)
    x_test = torch.stack([img for img, _ in test_ds]).to(device)
    y_test = torch.tensor([lbl for _, lbl in test_ds], device=device)

    y_train_oh = torch.zeros(len(y_train), 10, device=device)
    y_train_oh.scatter_(1, y_train.unsqueeze(1), 1.0)
    return x_train, y_train_oh, y_train, x_test, y_test


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

def evaluate(net, x_test, y_labels, batch_size=256, n_bins=15):
    """Test accuracy + calibration metrics (mirrors the Sequential example)."""
    net.eval()
    correct = 0
    conf_chunks, hit_chunks, nll_chunks, brier_chunks, out_var_chunks = [], [], [], [], []
    for i in range(0, len(x_test), batch_size):
        out = net(tensor(x_test[i : i + batch_size], var=0.0))
        mu, var = out.mu, out.var
        yb = y_labels[i : i + batch_size]
        conf, pred = mu.max(dim=1)
        hit = pred == yb
        correct += hit.sum().item()
        conf_chunks.append(conf)
        hit_chunks.append(hit.to(conf.dtype))

        p = mu.clamp(min=1e-12)
        p_true = p.gather(1, yb.view(-1, 1)).squeeze(1)
        nll_chunks.append(-p_true.log())

        onehot = torch.zeros_like(mu)
        onehot.scatter_(1, yb.view(-1, 1), 1.0)
        brier_chunks.append(((mu - onehot) ** 2).sum(dim=1))
        out_var_chunks.append(var.reshape(-1))

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
    }


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def make_sigma_v_schedule(start, end, decay_epochs, shape, n_epochs):
    """Return f(epoch:int)->sigma_v (linear/cosine/exp anneal, then hold)."""
    if end is None:
        return lambda epoch: start
    decay_epochs = min(decay_epochs, n_epochs)
    if decay_epochs <= 1:
        return lambda epoch: end

    def f(epoch):
        if epoch >= decay_epochs:
            return end
        t = (epoch - 1) / (decay_epochs - 1)
        if shape == "linear":
            return start + (end - start) * t
        if shape == "cosine":
            return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))
        if shape == "exp":
            return math.exp(math.log(start) + (math.log(end) - math.log(start)) * t)
        raise ValueError(f"unknown sigma_v schedule shape: {shape}")

    return f


def train(
    net, x_train, y_train_oh, x_test, y_test_labels,
    n_epochs, batch_size, sigma_v_fn, augment, device, run, config,
):
    """Training loop with optional GPU augmentation. Returns best test accuracy."""
    header = (
        f"\n  {'Epoch':>5}  {'σ_v':>6}  {'Acc':>7}  {'Conf':>6}  {'ECE':>6}"
        f"  {'NLL':>6}  {'Brier':>6}  {'oVarMax':>8}  {'Time':>7}"
    )
    print(header)
    print("  " + "─" * (len(header) - 3))

    best_acc = 0.0
    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        sv = sigma_v_fn(epoch)
        var_v = sv * sv  # autocov.observe() takes the observation-noise VARIANCE
        perm = torch.randperm(x_train.size(0), device=device)
        x_s, y_s = x_train[perm], y_train_oh[perm]

        for i in range(0, len(x_s), batch_size):
            xb = x_s[i : i + batch_size]
            if augment:
                xb = gpu_augment(xb)
            # Forward builds the graph; observe() runs the automatic backward
            # sweep and updates every parameter in place.
            out = net(tensor(xb, var=0.0))
            out.observe(y_s[i : i + batch_size], var_v=var_v)

        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0

        m = evaluate(net, x_test, y_test_labels)
        acc = m["test_acc"]
        best_acc = max(best_acc, acc)
        print(
            f"  {epoch:5d}  {sv:6.4f}  {acc*100:6.2f}%  {m['mean_conf']:6.3f}  {m['ece']:6.3f}"
            f"  {m['nll']:6.3f}  {m['brier']:6.3f}  {m['out_var_max']:8.2e}  {wall:6.2f}s",
            flush=True,
        )
        run.append_metrics(
            epoch, test_acc=acc, mean_conf=m["mean_conf"], ece=m["ece"], nll=m["nll"],
            brier=m["brier"], out_var_max=m["out_var_max"], out_var_mean=m["out_var_mean"],
            sigma_v=sv, wall_s=wall,
        )

    print("  " + "─" * 34)
    print(f"  Best test accuracy: {best_acc*100:.2f}%")
    return best_acc


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
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test accuracy (%)")
    ax.set_title("CIFAR-10 ResNet-18 (autocov) — TAGI test accuracy")
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
    augment: bool = True,
    data_dir: str = "data",
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    torch.manual_seed(seed)
    dev = torch.device(device)

    print("=" * 60)
    print("  CIFAR-10 ResNet-18 — triton-tagi (autocov)")
    print("  Stem+4 stages(64→128→256→512)+GAP → FC(512→10) → Remax")
    print("=" * 60)
    if device == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    print(f"\n  Loading CIFAR-10 from '{data_dir}'...", flush=True)
    x_train, y_train_oh, _, x_test, y_test_labels = load_cifar10(data_dir, dev)
    print(f"  Train: {x_train.shape[0]:,}  |  Test: {x_test.shape[0]:,}")

    config: dict = {
        "dataset": "cifar10", "arch": "resnet18", "optimizer": "tagi-autocov",
        "n_epochs": n_epochs, "batch_size": batch_size, "sigma_v": sigma_v,
        "sigma_v_end": sigma_v_end, "sigma_v_decay_epochs": sigma_v_decay_epochs,
        "sigma_v_schedule": sigma_v_schedule, "gain_w": gain_w, "gain_b": gain_b,
        "augment": augment, "seed": seed, "device": device, "engine": "autocov",
    }
    run = RunDir("cifar10", "resnet18", "tagi-autocov")
    run.save_config(config)
    print(f"  Run directory: {run.path}")

    # ── Network ──
    net = ResNet18(num_classes=10, gain_w=gain_w, gain_b=gain_b, device=device)
    print(f"  Parameters: {net.num_parameters:,}")
    sigma_v_fn = make_sigma_v_schedule(
        sigma_v, sigma_v_end, sigma_v_decay_epochs, sigma_v_schedule, n_epochs
    )
    sv_desc = (
        f"{sigma_v}" if sigma_v_end is None
        else f"{sigma_v}→{sigma_v_end} by ep{min(sigma_v_decay_epochs, n_epochs)} ({sigma_v_schedule})"
    )
    print(f"\n  Epochs: {n_epochs}  |  Batch: {batch_size}  |  σ_v: {sv_desc}  |  augment: {augment}")

    best_acc = train(
        net, x_train, y_train_oh, x_test, y_test_labels,
        n_epochs, batch_size, sigma_v_fn, augment, dev, run, config,
    )
    save_figure(run)
    print(f"\n  Results in: {run.path}")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 ResNet-18 (autocov) with TAGI")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sigma_v", type=float, default=0.05)
    parser.add_argument("--sigma_v_end", type=float, default=None)
    parser.add_argument("--sigma_v_decay_epochs", type=int, default=50)
    parser.add_argument("--sigma_v_schedule", choices=["linear", "cosine", "exp"], default="linear")
    parser.add_argument("--gain_w", type=float, default=0.1)
    parser.add_argument("--gain_b", type=float, default=0.1)
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.set_defaults(augment=True)
    args = parser.parse_args()
    main(**vars(args))
