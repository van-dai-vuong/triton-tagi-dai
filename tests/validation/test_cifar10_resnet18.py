"""Validation: triton-tagi ResNet-18 vs cuTAGI ResNet-18, weight-matched forward pass.

Level 1 — exact forward comparison:
  1. Build cuTAGI ResNet-18 with MixtureReLU (= triton's Bayesian ReLU: Gaussian moments).
  2. Initialize on GPU, run one eval-mode forward to populate GPU weights.
  3. Copy ALL cuTAGI weights into triton (handling Conv layout: (C_out,K)→(K,C_out)).
  4. Both networks in eval mode (BN uses initial running_mean=0, running_var=1 → identity).
  5. Forward on the same batch → compare (mu, var) within tolerance.

Architecture:
  Both: Conv(3→64,3,p=1,bias=T)+BN → 8 ResBlocks → GAP → Linear(512→11)
  cuTAGI: MixtureReLU, bias=False in ResBlock convs  (project shortcut included)
  triton:  ReLU (= MixtureReLU), bias=True in ResBlock convs (bias set to 0 to match cuTAGI)

Key layout facts:
  cuTAGI Conv mw: (C_out·K,) flat → triton: reshape(C_out,K).T → (K,C_out)
  cuTAGI BN  mw: (C,) flat gamma → triton: reshape(1,C)
  cuTAGI Linear mw: (C_out·C_in,) flat → triton: reshape(C_out,C_in).T → (C_in,C_out)

Note: cuTAGI has two ReLU variants:
  - ReLU: hard clip (mu_a = max(mu_z, 0)) — NOT used here
  - MixtureReLU: exact Gaussian moments via CDF/PDF — matches triton's ReLU exactly

Level 2 — accuracy comparison (2 epochs):
  Same architecture (MixtureReLU), independent random initializations.
  Pass criteria: both ≥ 20%, gap ≤ 15 pp.

Run with (Level 1 is fast; Level 2 takes ~6 min on RTX 4070):
    pytest tests/validation/test_cifar10_resnet18.py -v -s
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchvision import datasets, transforms

pytagi = pytest.importorskip("pytagi", reason="cuTAGI (pytagi) not installed")
from pytagi import HRCSoftmaxMetric, Utils
from pytagi.nn import AvgPool2d as PAvgPool2d
from pytagi.nn import BatchNorm2d as PBN
from pytagi.nn import Conv2d as PConv2d
from pytagi.nn import LayerBlock
from pytagi.nn import Linear as PLinear
from pytagi.nn import MixtureReLU as PMixReLU
from pytagi.nn import OutputUpdater
from pytagi.nn import ResNetBlock
from pytagi.nn import Sequential as PSequential

from triton_tagi import (
    AvgPool2D,
    BatchNorm2D,
    Conv2D,
    Flatten,
    Linear,
    ReLU,
    ResBlock,
    Sequential,
    class_to_obs,
    get_predicted_labels,
)

pytestmark = pytest.mark.cuda

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "data"
N_CLASSES = 10
HRC_LEN = 11

SIGMA_V = 0.1
BATCH = 128
BATCH_EVAL = 32
N_EPOCHS = 2
GAIN_W = 0.1
GAIN_B = 0.1

FWD_ATOL = 1e-3     # tolerance for weight-matched forward comparison

ACC_MIN = 0.20
ACC_TOL = 0.15

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


# ──────────────────────────────────────────────────────────────────────────────
#  Network builders
# ──────────────────────────────────────────────────────────────────────────────


def _main_block(in_c: int, out_c: int, stride: int = 1, padding_type: int = 1) -> LayerBlock:
    """cuTAGI ResBlock main path: Conv→ReLU→BN→Conv→ReLU→BN (bias=False in convs)."""
    return LayerBlock(
        PConv2d(in_c, out_c, 3, bias=False, stride=stride, padding=1,
                padding_type=padding_type, gain_weight=GAIN_W),
        PMixReLU(),
        PBN(out_c),
        PConv2d(out_c, out_c, 3, bias=False, padding=1, gain_weight=GAIN_W),
        PMixReLU(),
        PBN(out_c),
    )


def _build_pytagi() -> PSequential:
    """cuTAGI ResNet-18 with plain ReLU (same Gaussian moment propagation as triton)."""
    net = PSequential(
        PConv2d(3, 64, 3, bias=True, padding=1, in_width=32, in_height=32,
                gain_weight=GAIN_W),
        PMixReLU(), PBN(64),
        # Stage 1
        ResNetBlock(_main_block(64, 64)),
        ResNetBlock(_main_block(64, 64)),
        # Stage 2
        ResNetBlock(
            _main_block(64, 128, stride=2, padding_type=2),
            LayerBlock(PConv2d(64, 128, 2, bias=False, stride=2, gain_weight=GAIN_W),
                       PMixReLU(), PBN(128)),
        ),
        ResNetBlock(_main_block(128, 128)),
        # Stage 3
        ResNetBlock(
            _main_block(128, 256, stride=2, padding_type=2),
            LayerBlock(PConv2d(128, 256, 2, bias=False, stride=2, gain_weight=GAIN_W),
                       PMixReLU(), PBN(256)),
        ),
        ResNetBlock(_main_block(256, 256)),
        # Stage 4
        ResNetBlock(
            _main_block(256, 512, stride=2, padding_type=2),
            LayerBlock(PConv2d(256, 512, 2, bias=False, stride=2, gain_weight=GAIN_W),
                       PMixReLU(), PBN(512)),
        ),
        ResNetBlock(_main_block(512, 512)),
        PAvgPool2d(4),
        PLinear(512, HRC_LEN, gain_weight=GAIN_W, gain_bias=GAIN_B),
    )
    net.preinit_layer()
    net.to_device("cuda")
    return net


def _build_triton() -> Sequential:
    kw = {"device": DEVICE, "gain_w": GAIN_W, "gain_b": GAIN_B}
    return Sequential(
        [
            Conv2D(3, 64, 3, stride=1, padding=1, **kw),
            ReLU(),
            BatchNorm2D(64, preserve_var=False, **kw),
            ResBlock(64, 64, stride=1, **kw),
            ResBlock(64, 64, stride=1, **kw),
            ResBlock(64, 128, stride=2, **kw),
            ResBlock(128, 128, stride=1, **kw),
            ResBlock(128, 256, stride=2, **kw),
            ResBlock(256, 256, stride=1, **kw),
            ResBlock(256, 512, stride=2, **kw),
            ResBlock(512, 512, stride=1, **kw),
            AvgPool2D(4),
            Flatten(),
            Linear(512, HRC_LEN, **kw),
        ],
        device=DEVICE,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Weight-copy helpers  (cuTAGI → triton layout conversions)
# ──────────────────────────────────────────────────────────────────────────────


def _load_conv(triton_conv: Conv2D, sd: dict, key: str, has_bias: bool) -> None:
    """Copy cuTAGI conv weights (C_out, K) flat → triton (K, C_out)."""
    mw_flat = np.array(sd[key][0])
    Sw_flat = np.array(sd[key][1])
    K, C_out = triton_conv.mw.shape
    assert len(mw_flat) == K * C_out, f"{key}: expected {K*C_out} got {len(mw_flat)}"
    triton_conv.mw = torch.tensor(
        mw_flat.reshape(C_out, K).T, dtype=torch.float32, device=DEVICE)
    triton_conv.Sw = torch.tensor(
        Sw_flat.reshape(C_out, K).T, dtype=torch.float32, device=DEVICE)
    if has_bias:
        mb_flat = np.array(sd[key][2])
        Sb_flat = np.array(sd[key][3])
        triton_conv.mb = torch.tensor(
            mb_flat.reshape(1, C_out), dtype=torch.float32, device=DEVICE)
        triton_conv.Sb = torch.tensor(
            Sb_flat.reshape(1, C_out), dtype=torch.float32, device=DEVICE)
    else:
        triton_conv.mb = torch.zeros_like(triton_conv.mb)
        triton_conv.Sb = torch.zeros_like(triton_conv.Sb)


def _load_bn(triton_bn: BatchNorm2D, sd: dict, key: str) -> None:
    """Copy cuTAGI BN params (C,) flat → triton (1, C)."""
    gamma = np.array(sd[key][0])
    Sgamma = np.array(sd[key][1])
    beta = np.array(sd[key][2])
    Sbeta = np.array(sd[key][3])
    C = len(gamma)
    triton_bn.mw = torch.tensor(gamma.reshape(1, C), dtype=torch.float32, device=DEVICE)
    triton_bn.Sw = torch.tensor(Sgamma.reshape(1, C), dtype=torch.float32, device=DEVICE)
    triton_bn.mb = torch.tensor(beta.reshape(1, C), dtype=torch.float32, device=DEVICE)
    triton_bn.Sb = torch.tensor(Sbeta.reshape(1, C), dtype=torch.float32, device=DEVICE)


def _load_linear(triton_lin: Linear, sd: dict, key: str) -> None:
    """Copy cuTAGI Linear weights (C_out, C_in) flat → triton (C_in, C_out)."""
    mw_flat = np.array(sd[key][0])
    Sw_flat = np.array(sd[key][1])
    mb_flat = np.array(sd[key][2])
    Sb_flat = np.array(sd[key][3])
    C_in, C_out = triton_lin.mw.shape
    triton_lin.mw = torch.tensor(
        mw_flat.reshape(C_out, C_in).T, dtype=torch.float32, device=DEVICE)
    triton_lin.Sw = torch.tensor(
        Sw_flat.reshape(C_out, C_in).T, dtype=torch.float32, device=DEVICE)
    triton_lin.mb = torch.tensor(
        mb_flat.reshape(1, C_out), dtype=torch.float32, device=DEVICE)
    triton_lin.Sb = torch.tensor(
        Sb_flat.reshape(1, C_out), dtype=torch.float32, device=DEVICE)


def _sync_weights(net_tri: Sequential, net_cut: PSequential) -> None:
    """
    Copy cuTAGI GPU weights → triton after params_to_host().

    cuTAGI Sequential layout (indices match both frameworks):
      0: Conv2d (stem, bias=True)   1: ReLU   2: BN
      3..10: ResNetBlock             11: AvgPool  12: Linear
    Triton layers[]: same indices 0-10, then 11=AvgPool, 12=Flatten, 13=Linear
    """
    net_cut.params_to_host()
    sd = net_cut.state_dict()  # keys have 'Cuda' suffix after to_device

    layers = net_tri.layers

    # Stem
    _load_conv(layers[0], sd, "Conv2dCuda.0", has_bias=True)
    _load_bn(layers[2], sd, "BatchNorm2dCuda.2")

    # 8 ResBlocks at cuTAGI positions 3..10 = triton layers[3..10]
    proj_positions = {5, 7, 9}   # ResBlocks that have projection shortcuts
    for pos in range(3, 11):
        blk: ResBlock = layers[pos]
        p = str(pos)
        _load_conv(blk.conv1, sd, f"Conv2dCuda.main.{p}.0", has_bias=False)
        _load_bn(blk.bn1, sd, f"BatchNorm2dCuda.main.{p}.2")
        _load_conv(blk.conv2, sd, f"Conv2dCuda.main.{p}.3", has_bias=False)
        _load_bn(blk.bn2, sd, f"BatchNorm2dCuda.main.{p}.5")
        if pos in proj_positions:
            _load_conv(blk.proj_conv, sd, f"Conv2dCuda.shortcut.{p}.0", has_bias=False)
            _load_bn(blk.proj_bn, sd, f"BatchNorm2dCuda.shortcut.{p}.2")

    # Head Linear (cuTAGI pos 12 = triton layers[13])
    _load_linear(layers[13], sd, "LinearCuda.12")


# ──────────────────────────────────────────────────────────────────────────────
#  Data
# ──────────────────────────────────────────────────────────────────────────────


def _load_cifar10():
    tf = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    train_ds = datasets.CIFAR10(DATA_ROOT, train=True, download=False, transform=tf)
    test_ds = datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=tf)
    x_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    y_train = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
    x_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
    y_test = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
    return x_train, y_train, x_test, y_test


# ──────────────────────────────────────────────────────────────────────────────
#  Level 1: forward output matches with shared weights
# ──────────────────────────────────────────────────────────────────────────────


def test_cifar10_resnet18_forward_matched_weights():
    """After weight sync, triton and cuTAGI ResNet-18 produce the same output.

    Both networks run in eval mode with initial BN state (running_mean=0,
    running_var=1, gamma=1, beta=0) → BN acts as near-identity.
    With identical Conv/Linear weights, outputs must agree within 1e-3.

    Order: cuTAGI runs on test batch first (saving output), THEN params_to_host()
    is called to sync weights to triton. Calling net_cut after params_to_host()
    would invalidate the cuTAGI CUDA state.
    """
    torch.manual_seed(42)
    x_batch = torch.randn(8, 3, 32, 32, device=DEVICE)
    x_np = x_batch.cpu().numpy().reshape(-1).astype(np.float32)

    net_cut = _build_pytagi()
    net_tri = _build_triton()

    # Run cuTAGI in eval mode — no throw-away; both networks start from the
    # identical initial BN state (running_mean=0, running_var=1 → identity).
    net_cut.eval()

    # --- Run cuTAGI on test batch BEFORE params_to_host ---
    mu_cut_flat, var_cut_flat = net_cut(x_np)
    mu_cut = torch.tensor(mu_cut_flat, dtype=torch.float32).reshape(8, HRC_LEN)
    var_cut = torch.tensor(var_cut_flat, dtype=torch.float32).reshape(8, HRC_LEN)

    # --- Sync cuTAGI GPU weights → triton (params_to_host invalidates cuTAGI CUDA) ---
    _sync_weights(net_tri, net_cut)

    # --- Run triton on same batch (eval, BN at initial identity state) ---
    net_tri.eval()
    with torch.no_grad():
        mu_tri, var_tri = net_tri.forward(x_batch)

    mu_diff = (mu_tri.cpu() - mu_cut).abs().max().item()
    var_diff = (var_tri.cpu() - var_cut).abs().max().item()

    print(f"\n  max |Δmu|  = {mu_diff:.2e}  (tol {FWD_ATOL:.0e})")
    print(f"  max |Δvar| = {var_diff:.2e}  (tol {FWD_ATOL:.0e})")

    assert mu_diff < FWD_ATOL, f"mean mismatch {mu_diff:.2e} > {FWD_ATOL}"
    assert var_diff < FWD_ATOL, f"var  mismatch {var_diff:.2e} > {FWD_ATOL}"


# ──────────────────────────────────────────────────────────────────────────────
#  Level 2: accuracy comparison (2 epochs, independent random inits)
# ──────────────────────────────────────────────────────────────────────────────


def test_cifar10_resnet18_hrc_2epochs():
    """Both reach ≥ 20 % and are within 15 % of each other.

    Networks are trained and evaluated sequentially (sequential = less peak GPU memory).
    Takes ~6 minutes on RTX 4070.
    """
    torch.manual_seed(0)

    tri_hrc = class_to_obs(N_CLASSES)
    utils = Utils()
    metric = HRCSoftmaxMetric(num_classes=N_CLASSES)

    x_train, y_train, x_test, y_test = _load_cifar10()

    # ── triton-tagi ──────────────────────────────────────────────────────────
    net_tri = _build_triton()
    torch.manual_seed(1)
    for _epoch in range(N_EPOCHS):
        perm = torch.randperm(len(x_train))
        x_s, y_s = x_train[perm], y_train[perm]
        net_tri.train()
        for i in range(0, len(x_s), BATCH):
            net_tri.step_hrc(
                x_s[i : i + BATCH].to(DEVICE), y_s[i : i + BATCH].to(DEVICE),
                tri_hrc, SIGMA_V,
            )

    net_tri.eval()
    correct_tri = 0
    x_test_gpu = x_test.to(DEVICE)
    with torch.no_grad():
        for i in range(0, len(x_test_gpu), BATCH_EVAL):
            mu, Sa = net_tri.forward(x_test_gpu[i : i + BATCH_EVAL])
            preds = get_predicted_labels(mu, Sa, tri_hrc)
            correct_tri += (preds.cpu() == y_test[i : i + BATCH_EVAL]).sum().item()
    acc_tri = correct_tri / len(y_test)

    del net_tri, x_test_gpu
    torch.cuda.empty_cache()

    # ── cuTAGI ───────────────────────────────────────────────────────────────
    net_cut = _build_pytagi()
    updater = OutputUpdater(net_cut.device)
    torch.manual_seed(2)
    for _epoch in range(N_EPOCHS):
        perm = torch.randperm(len(x_train))
        x_np = x_train[perm].numpy()
        y_np = y_train[perm].numpy().astype(np.int32)
        for i in range(0, len(x_np), BATCH):
            xb_np = x_np[i : i + BATCH]
            lb_np = y_np[i : i + BATCH]
            nb = len(lb_np)
            obs_np, obs_idx_np, _ = utils.label_to_obs(lb_np, N_CLASSES)
            var_yb = np.full(nb * tri_hrc.n_obs, SIGMA_V**2, dtype=np.float32)
            net_cut(xb_np.reshape(-1).astype(np.float32))
            updater.update_using_indices(
                output_states=net_cut.output_z_buffer,
                mu_obs=obs_np.astype(np.float32),
                var_obs=var_yb,
                selected_idx=obs_idx_np.astype(np.int32),
                delta_states=net_cut.input_delta_z_buffer,
            )
            net_cut.backward()
            net_cut.step()

    correct_cut = 0
    x_test_np = x_test.numpy()
    for i in range(0, len(x_test_np), BATCH_EVAL):
        xb_np = x_test_np[i : i + BATCH_EVAL]
        nb = len(xb_np)
        ma_flat, Sa_flat = net_cut(xb_np.reshape(-1).astype(np.float32))
        preds = metric.get_predicted_labels(np.array(ma_flat), np.array(Sa_flat))
        correct_cut += (torch.tensor(preds, dtype=torch.long) == y_test[i : i + nb]).sum().item()
    acc_cut = correct_cut / len(y_test)

    print(f"\n  triton-tagi ResNet-18 HRC:  {acc_tri * 100:.2f}%")
    print(f"  cuTAGI ResNet-18 (ReLU):    {acc_cut * 100:.2f}%")
    print(f"  Δ accuracy:                 {abs(acc_tri - acc_cut) * 100:.3f}%"
          f"  (tol {ACC_TOL * 100:.1f}%)")

    assert acc_tri >= ACC_MIN, f"triton-tagi {acc_tri * 100:.2f}% < {ACC_MIN * 100:.0f}%"
    assert acc_cut >= ACC_MIN, f"cuTAGI {acc_cut * 100:.2f}% < {ACC_MIN * 100:.0f}%"
    assert abs(acc_tri - acc_cut) < ACC_TOL, (
        f"gap {abs(acc_tri - acc_cut) * 100:.3f}% > {ACC_TOL * 100:.1f}%  "
        f"tri={acc_tri * 100:.2f}%  cut={acc_cut * 100:.2f}%"
    )
