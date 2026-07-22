# triton-tagi

**Tractable Approximate Gaussian Inference (TAGI) for Bayesian Neural Networks, in pure PyTorch.**

A minimal reimplementation of [cuTAGI](https://github.com/lhnguyen102/cuTAGI)
(C++/CUDA) with numerical parity on its headline examples. It runs on **CPU**
(including Apple Silicon Macs) out of the box, and on CUDA/MPS when available.

> **Note.** This started as a [Triton](https://triton-lang.org/)-kernel port
> and was subsequently rewritten to pure PyTorch tensor ops so it runs anywhere
> PyTorch does — no Triton, no CUDA required. The math is unchanged; the hot
> paths (fused variance / backward-delta matmuls, im2col via `unfold`/`fold`)
> are expressed as vectorised `torch` operations.

> **Idea.** TAGI treats every weight and activation as a Gaussian. The forward
> pass propagates `(mean, variance)` analytically through each layer; the
> backward pass applies closed-form Bayesian updates to the parameters. No
> sampling, no variational bounds, no autograd.

---

## Status

- **Library version:** 0.2.0 (scoped down 2026-04-19 to a minimal cuTAGI-parity core).
- **Parity:** every kept example reproduces its cuTAGI counterpart at Phase-1
  tolerance; see [`PLAN.md`](PLAN.md) §3 for the table.
- **Tests:** 101 unit tests pass on CPU; the validation suite (numerical parity
  vs the `pytagi` reference) is marked `cuda` and runs on a GPU.
- **Archive:** layers, optimizers, and diagnostics outside the minimal scope
  live under [`_archive/`](_archive/); nothing deleted.

### Recent milestones

- **2026-04-23.** Per-call dispatch overhead in `kernels/attention.py` cut
  17–24% by replacing `@triton.autotune` with a shape-adaptive `_pick_blocks`
  heuristic; attention validation suite (18 tests) still passes. CIFAR-10
  ResNet-18 + Remax reaches **89%** test accuracy after exact cuTAGI parity.
- **2026-04-22.** Self-attention added (`Embedding`, `PositionalEncoding`,
  `MultiheadAttentionV2`, `RMSNorm`) plus the `reverse_predictor` example —
  sequence reversal with sinusoidal PE + MHA-V2 + RMSNorm + HRC head,
  matching cuTAGI's `feat/attn-debug` branch.
- **2026-04-20.** Remax reimplemented to match cuTAGI's MixtureReLU plus
  log-normal covariance path. CIFAR-10 ResNet-18 + Remax now trains to ≥80%
  test accuracy in ~15 epochs with `gain_w=gain_b=0.1` and σ_v ∈ {0.01, 0.05}.

---

## Install

```bash
git clone https://github.com/miquelflorensa/triton-tagi.git
cd triton-tagi
pip install -e .           # core: torch, numpy
pip install -e ".[vis]"    # + matplotlib for figures
pip install -e ".[dev]"    # + pytest, ruff
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0 (CPU build is fine). No CUDA or
Triton dependency.

---

## Quick start

```python
import torch
from triton_tagi import Linear, ReLU, Remax, Sequential

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Sequential(
    [
        Linear(784, 256, device=device),
        ReLU(),
        Linear(256, 128, device=device),
        ReLU(),
        Linear(128, 10, device=device),
        Remax(),
    ],
    device=device,
)

# One closed-form Bayesian update step
y_pred_mu, y_pred_var = net.step(x_batch, y_batch_onehot, sigma_v=0.05)
```

Every example under [`examples/`](examples/) follows the same `RunDir` and
`argparse` convention (see [PLAN.md](PLAN.md) §3.1):

```bash
python examples/mnist_mlp.py --n_epochs 5
python examples/mnist_cnn.py
python examples/cifar10_cnn.py --n_epochs 100
python examples/cifar10_resnet18.py --n_epochs 100 --gain_w 0.1 --gain_b 0.1
python examples/cifar10_resnet18_hrc.py
python examples/regression.py
python examples/regression_heteros.py
python examples/reverse_predictor.py    # sinusoidal PE + MHA-V2 + RMSNorm + HRC head
python examples/custom_layer.py         # tutorial: write your own Triton layer
```

---

## Library surface

Everything lives under `triton_tagi/`. The package is deliberately small;
reading it top-to-bottom in an evening is a goal, not an accident.

### Layers (`triton_tagi/layers/`)

| Layer | Used by |
|---|---|
| `Linear` | MLP, CNN/ResNet heads |
| `Conv2D` | CNN, ResNet |
| `BatchNorm2D` | CNN, ResNet |
| `LayerNorm` | MLP variant |
| `AvgPool2D`, `MaxPool2D` | CNN, ResNet stem/head |
| `ReLU` | all non-linear examples |
| `Flatten` | conv→FC boundary |
| `Remax` | classification head (cuTAGI-native) |
| `ResBlock` + `Add` | ResNet-18 |
| `EvenSoftplus` | heteroscedastic regression noise head |
| `Embedding` | reverse_predictor (token → vector) |
| `PositionalEncoding` | reverse_predictor (sinusoidal, fixed) |
| `MultiheadAttentionV2` | reverse_predictor (separate Q/K/V projections, Remax over scores) |
| `RMSNorm` | reverse_predictor |

### Top-level

- `base.py`: `Layer`, `LearnableLayer` ABCs.
- `network.py`: `Sequential` (forward / step / train / eval).
- `param_init.py`: He / Xavier / Gaussian init.
- `hrc_softmax.py`: hierarchical softmax output for many-class classification.
- `checkpoint.py`: `RunDir` (run-directory manager) and `load_model`.
- `kernels/common.py`: vectorised `torch` matmuls for Linear / Conv2D / BN
  (fused variance forward + backward-delta).
- `kernels/attention.py`: batched `torch` matmuls for `MultiheadAttentionV2`
  (`bmm_tagi_var` for the QKᵀ / Score@V variance, `bmm_shared_left/right`
  for the four backward reductions).
- `update/observation.py`, `update/parameters.py`: innovation and parameter update rules.

---

## How TAGI works (one page)

Every weight `W` and activation `a` is a Gaussian random variable described
by its mean `μ` and variance `σ²`.

**Forward (moment propagation).** For `z = a W + b`,

$$\mu_z = \mu_a\,\mu_W + \mu_b$$

$$\sigma^2_z = \mu_a^2\,\sigma^2_W + \sigma^2_a\,\mu_W^2 + \sigma^2_a\,\sigma^2_W + \sigma^2_b$$

Nonlinearities propagate moments analytically. ReLU uses the Alric (2024)
closed-form; Remax uses MixtureReLU plus log-normal identities, matching cuTAGI.

**Backward (observation innovation).** At the output,

$$\delta_\mu = \frac{y - \mu_z}{\sigma^2_z + \sigma^2_v}, \qquad \delta_\sigma = \frac{-1}{\sigma^2_z + \sigma^2_v}.$$

Deltas propagate backward through each layer; no autograd.

**Parameter update (capped, cuTAGI-style).**

$$\mu_W^{\text{new}} = \mu_W + \sigma^2_W \cdot \Delta_\mu$$

$$\sigma^{2,\text{new}}_W = \max\!\left(\sigma^2_W + (\sigma^2_W)^2 \cdot \Delta_\sigma,\;\epsilon\right)$$

---

## Writing a custom layer

Subclass `Layer` (pure moment propagation) or `LearnableLayer` (adds `update`
and `num_parameters`), implement `forward` and `backward`, and you are done.
No registry, no decorators. See [`examples/custom_layer.py`](examples/custom_layer.py)
for an end-to-end ELU tutorial: `Layer` subclass and MNIST run.

---

## Tests

```bash
pytest tests/unit                 # ~95 tests, fast
pytest tests/validation           # ~89 tests; compares to pytagi reference
pytest -m "not slow and not cuda" # CPU subset
```

Validation tests assert `torch.testing.assert_close(atol=1e-5, rtol=0)`
against a pytagi reference run on the same batch.

---

## Benchmarks

> **Historical.** The numbers below are from the original Triton/GPU build and
> are kept for reference; the pure-PyTorch port targets portability (CPU/Mac)
> rather than peak GPU throughput.

See [`benchmarks/results.md`](benchmarks/results.md). Summary on an RTX 4070
Ti SUPER, median of 50 runs:

| Layer / batch 1024 | triton-tagi | cuTAGI | Speedup |
|---|---:|---:|---:|
| Linear(512, 512) | 0.64 ms | 45.0 ms | **70×** |
| Conv2D net | 11.0 ms | 106 ms | **9.7×** |
| BatchNorm2D net | 12.5 ms | 109 ms | **8.7×** |

triton-tagi wins on throughput; cuTAGI wins on small-batch latency, where
dispatch overhead dominates.

---

## Relation to cuTAGI

This is a Triton-based reimplementation of
[cuTAGI](https://github.com/lhnguyen102/cuTAGI), the reference C++/CUDA TAGI
library. Parity is load-bearing: every kept example must reproduce the cuTAGI
result at Phase-1 tolerance on a fixed seed. In particular:

- **Capped parameter updates** with batch-size-dependent cap factors.
- **Backward order:** compute deltas first, apply capped updates after.
- **ResBlock** identical to cuTAGI's `ResNetBlock` (projection shortcut).
- **Remax** uses cuTAGI's MixtureReLU plus log-normal covariance path, not a
  Softplus+Taylor approximation. See `triton_tagi/layers/remax.py`.

TF32 matmul is disabled at import time (a no-op on CPU-only builds): cuTAGI
uses scalar FMA with near-fp64 accuracy, so leaving TF32 on when running on
CUDA would introduce systematic ~1e-3 variance errors and break parity.

---

## References

- Goulet, J.-A., Nguyen, L. H., & Amiri, S. (2021). *Tractable Approximate
  Gaussian Inference for Bayesian Neural Networks*. JMLR 22(228), 1–23.
  [[paper]](https://www.jmlr.org/papers/v22/20-1009.html)
- Alric, L. (2024). Closed-form MixtureReLU moments.
- cuTAGI: [github.com/lhnguyen102/cuTAGI](https://github.com/lhnguyen102/cuTAGI).
- Triton: [triton-lang.org](https://triton-lang.org/).
