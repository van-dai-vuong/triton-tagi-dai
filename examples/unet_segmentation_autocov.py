"""
U-Net image segmentation with a Bayesian U-Net (autocov).

Builds the classic U-shaped encoder–decoder with skip connections
(https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/)
entirely from autocov ops, and trains it on a self-contained synthetic task:
segment a bright disk out of a noisy background. Per-pixel sigmoid output +
``observe`` gives a segmentation mask *and* a per-pixel uncertainty map — no
optimizer / backward code.

    encoder :  ConvBlock → pool → ConvBlock → pool → bottleneck   (contracting)
    decoder :  upsample → concat(skip) → ConvBlock  (×2)          (expanding)
    head    :  Conv(→1) → sigmoid                                 (mask prob)

The skip connections are ``concat`` (encoder feature ⧺ decoder feature); the
decoder upsamples with ``upsample``. Both are exact autocov ops.

Usage:
    python examples/unet_segmentation_autocov.py
    python examples/unet_segmentation_autocov.py --n_epochs 60 --size 16 --noise 0.3
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from triton_tagi.autocov import (
    BatchNorm2D,
    Conv2D,
    MaxPool2D,
    Module,
    concat,
    relu,
    sigmoid,
    tensor,
    upsample,
)


# ---------------------------------------------------------------------------
#  Building block:  Conv(3×3) → ReLU → BatchNorm
# ---------------------------------------------------------------------------

class ConvBlock(Module):
    def __init__(self, c_in, c_out, device=None, gain_w=0.5, gain_b=0.5):
        super().__init__()
        self.conv = Conv2D(c_in, c_out, 3, stride=1, padding=1,
                           device=device, gain_w=gain_w, gain_b=gain_b)
        self.bn = BatchNorm2D(c_out, preserve_var=False,
                              device=device, gain_w=gain_w, gain_b=gain_b)

    def forward(self, x):
        return self.bn(relu(self.conv(x)))


# ---------------------------------------------------------------------------
#  U-Net
# ---------------------------------------------------------------------------

class UNet(Module):
    """A small U-Net: 2 downsampling stages, a bottleneck, 2 upsampling stages
    with concat skip connections, and a 1×1 conv + sigmoid mask head."""

    def __init__(self, base: int = 8, device=None, gain_w=0.5, gain_b=0.5):
        super().__init__()
        kw = dict(device=device, gain_w=gain_w, gain_b=gain_b)
        self.enc1 = ConvBlock(1, base, **kw)               # H
        self.enc2 = ConvBlock(base, 2 * base, **kw)        # H/2
        self.bott = ConvBlock(2 * base, 4 * base, **kw)    # H/4
        self.dec2 = ConvBlock(4 * base + 2 * base, 2 * base, **kw)  # H/2 (concat enc2)
        self.dec1 = ConvBlock(2 * base + base, base, **kw)         # H   (concat enc1)
        self.head = Conv2D(base, 1, 1, stride=1, padding=0, **kw)  # 1×1 → mask logit
        self.pool = MaxPool2D(2)

    def forward(self, x):
        e1 = self.enc1(x)                       # (B, base,   H,   W)
        e2 = self.enc2(self.pool(e1))           # (B, 2base,  H/2, W/2)
        b = self.bott(self.pool(e2))            # (B, 4base,  H/4, W/4)
        d2 = self.dec2(concat(upsample(b, 2), e2))   # (B, 2base, H/2, W/2)
        d1 = self.dec1(concat(upsample(d2, 2), e1))  # (B, base,  H,   W)
        return sigmoid(self.head(d1))           # (B, 1, H, W) mask probability


# ---------------------------------------------------------------------------
#  Synthetic data: a bright disk on a noisy background; target = its mask
# ---------------------------------------------------------------------------

def make_data(n: int, size: int = 16, noise: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    imgs = np.zeros((n, 1, size, size), np.float32)
    masks = np.zeros((n, 1, size, size), np.float32)
    for i in range(n):
        cy, cx = rng.uniform(size * 0.3, size * 0.7, size=2)
        r = rng.uniform(size * 0.15, size * 0.30)
        disk = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
        masks[i, 0] = disk.astype(np.float32)
        imgs[i, 0] = disk + rng.normal(0.0, noise, size=(size, size))
    return imgs, masks


def metrics(prob: np.ndarray, mask: np.ndarray):
    pred = (prob > 0.5).astype(np.float32)
    acc = float((pred == mask).mean())
    inter = float((pred * mask).sum())
    union = float(((pred + mask) > 0).sum())
    iou = inter / union if union > 0 else 1.0
    return acc, iou


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(
    n_epochs: int = 60,
    n_train: int = 64,
    n_test: int = 32,
    size: int = 16,
    base: int = 8,
    noise: float = 0.3,
    sigma_v: float = 0.2,
    gain_w: float = 0.5,
    gain_b: float = 0.5,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    torch.manual_seed(seed)
    dev = torch.device(device)
    print("=" * 60)
    print("  U-Net segmentation — Bayesian U-Net (autocov)")
    print("=" * 60)

    # ── Data ──
    Xtr, Ytr = make_data(n_train, size, noise, seed=seed)
    Xte, Yte = make_data(n_test, size, noise, seed=seed + 1)
    m, s = Xtr.mean(), Xtr.std() + 1e-8
    xtr = torch.tensor((Xtr - m) / s, device=dev)
    ytr = torch.tensor(Ytr, device=dev)
    xte = torch.tensor((Xte - m) / s, device=dev)
    print(f"  images: {size}×{size}  |  noise σ={noise}  |  train {n_train}  test {n_test}")

    # ── Model ──
    net = UNet(base=base, device=device, gain_w=gain_w, gain_b=gain_b)
    net.assign_names()
    print(f"  U-Net(base={base}): enc→pool→enc→pool→bottleneck→up+skip→up+skip→1×1")
    print(f"  parameters: {net.num_parameters:,}")

    # ── Train (full batch) ──
    print(f"\n  {'Epoch':>5}  {'TrainAcc':>8}  {'TestAcc':>8}  {'TestIoU':>8}  {'Time':>7}")
    print("  " + "─" * 46)
    var_v = sigma_v ** 2
    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        out = net(tensor(xtr, var=0.0))          # forward builds the graph
        tr_acc, _ = metrics(out.mu.detach().cpu().numpy(), Ytr)
        out.observe(ytr, var_v=var_v)            # automatic backward + update
        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            net.eval()
            pte = net(tensor(xte, var=0.0)).mu.detach().cpu().numpy()
            net.train()
            te_acc, te_iou = metrics(pte, Yte)
            print(f"  {epoch:5d}  {tr_acc*100:7.2f}%  {te_acc*100:7.2f}%  {te_iou:8.3f}"
                  f"  {time.perf_counter()-t0:6.2f}s", flush=True)

    # ── Final test + uncertainty ──
    net.eval()
    out = net(tensor(xte, var=0.0))
    out.print_graph()
    net.train()
    prob = out.mu.detach().cpu().numpy()
    unc = np.sqrt(out.var.detach().cpu().numpy() + var_v)   # predictive std per pixel
    acc, iou = metrics(prob, Yte)
    print("  " + "─" * 46)
    print(f"  Final test pixel-accuracy: {acc*100:.2f}%   IoU: {iou:.3f}")
    print(f"  Per-pixel predictive std: mean {unc.mean():.3f}, "
          f"max {unc.max():.3f}  (highest near object boundaries)")

    # ── Optional figure: input | truth | prediction | uncertainty ──
    try:
        import matplotlib.pyplot as plt
        k = 4
        fig, ax = plt.subplots(k, 4, figsize=(8, 2 * k))
        cols = ["input (noisy)", "true mask", "pred prob", "uncertainty (σ)"]
        for i in range(k):
            for j, img in enumerate([Xte[i, 0], Yte[i, 0], prob[i, 0], unc[i, 0]]):
                ax[i, j].imshow(img, cmap="viridis"); ax[i, j].axis("off")
                if i == 0:
                    ax[i, j].set_title(cols[j], fontsize=9)
        fig.tight_layout(); fig.savefig("unet_segmentation.png", dpi=140); plt.close(fig)
        print("  Figure saved to unet_segmentation.png")
    except ImportError:
        print("  (matplotlib not installed — skipping figure)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bayesian U-Net segmentation (autocov)")
    p.add_argument("--n_epochs", type=int, default=60)
    p.add_argument("--n_train", type=int, default=64)
    p.add_argument("--n_test", type=int, default=32)
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--base", type=int, default=8)
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--sigma_v", type=float, default=0.2)
    p.add_argument("--gain_w", type=float, default=0.5)
    p.add_argument("--gain_b", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    main(**vars(p.parse_args()))
