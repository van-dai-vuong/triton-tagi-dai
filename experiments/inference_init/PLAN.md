# Inference-Based Initialization (IBI) — Plan

Status: **draft 2026-04-23** — algorithm derived from Contribution 3 of the
user's thesis. Re-implementing from scratch (existing archived code in
`_archive/triton_tagi/inference_init.py` deliberately not consulted).

This is a research track, not part of the library's parity goal. It lives
under `experiments/` so it can grow figures, sweeps, and notebooks without
bloating the main library plan.

---

## 1. Goal

Replace He initialization in TAGI training with a **pre-training calibration
phase** that, for each layer $l$, drives the empirical hidden-unit moments
$(\mu_{Z_i}, \sigma_{Z_i}^2)$ toward a prescribed target $(\sigma_M, \sigma_Z)$
using one epoch of forward-and-correct passes over the data.

Acceptance for V1: reproduce Figure 4 of the thesis on MNIST+MLP.
Specifically:

- $\mathtt{L}=7$, $\sigma_V=0.01$, $(\sigma_M=1.0, \sigma_Z=0.5)$ → ~96.8%
  (vs He → 14.5%)
- $\mathtt{L}=5$, $\sigma_V=0.05$, $(\sigma_M=0.5, \sigma_Z=0.5)$ → ~97.7%
  (vs He → collapse)
- He baseline: matches the thesis numbers within run-to-run variance

---

## 2. Algorithm (one batch, one layer)

Notation: $A^{(l)}$ = layer $l$ width. $Z_i^{(l)}$ = pre-activation unit $i$.
Targets are derived from $(\sigma_M, \sigma_Z)$ (global hyperparameters).

### 2.1 Per-layer targets (depend only on $A$, $\sigma_M$, $\sigma_Z$)

$$
\mu_{\tilde S} = 0,\quad \sigma_{\tilde S}^2 = A\,\sigma_Z^2
$$
$$
\mu_{\tilde{S2}} = A(\sigma_M^2 + \sigma_Z^2),\quad
\sigma_{\tilde{S2}}^2 = A(2\sigma_Z^4 + 4\sigma_M^2\sigma_Z^2)
$$

### 2.2 Forward to layer $l$

Get current $(\mu_{Z_i}, \sigma_{Z_i}^2)$ from a TAGI forward pass on the
current batch through (already-calibrated) layers $1..l-1$ then layer $l$.

### 2.3 S projection (moment-matching, diagonal innovation)

$$
\mu_S = \sum_i \mu_{Z_i},\quad \sigma_S^2 = \sum_i \sigma_{Z_i}^2
$$
$$
\delta\mu_S = (\mu_{\tilde S} - \mu_S)/\sigma_S^2,\quad
\delta\sigma_S^2 = (\sigma_{\tilde S}^2 - \sigma_S^2)/\sigma_S^4
$$
$$
\mu_{Z_i|S} = \mu_{Z_i} + \sigma_{Z_i}^2 \delta\mu_S,\quad
\sigma_{Z_i|S}^2 = \sigma_{Z_i}^2 (1 + \sigma_S^2 \delta\sigma_S^2)
$$

### 2.4 S2 RTS update (quadratic obs, applied AFTER S)

Using the post-S moments as the new $\mu_{Z_i}, \sigma_{Z_i}^2$:

$$
\mu_{Z_i^2} = \mu_{Z_i}^2 + \sigma_{Z_i}^2,\quad
\sigma_{Z_i^2}^2 = 2\sigma_{Z_i}^4 + 4\sigma_{Z_i}^2\mu_{Z_i}^2
$$
$$
\mu_{S2} = \sum_i \mu_{Z_i^2},\quad \sigma_{S2}^2 = \sum_i \sigma_{Z_i^2}^2,\quad
J_i = 2\mu_{Z_i}\sigma_{Z_i}^2 / \sigma_{S2}^2
$$
$$
\delta\mu_{S2} = (\mu_{\tilde{S2}} - \mu_{S2})/\sigma_{S2}^2,\quad
\delta\sigma_{S2}^2 = (\sigma_{\tilde{S2}}^2 - \sigma_{S2}^2)/\sigma_{S2}^4
$$
$$
\mu_{Z_i|S2} = \mu_{Z_i} + 2\mu_{Z_i}\sigma_{Z_i}^2 \delta\mu_{S2},\quad
\sigma_{Z_i|S2}^2 = \sigma_{Z_i}^2 + (2\mu_{Z_i}\sigma_{Z_i}^2)^2 \delta\sigma_{S2}^2
$$

After 2.3 + 2.4 we have the calibrated targets $(\mu_{Z_i|\cdot}, \sigma_{Z_i|\cdot}^2)$.

### 2.5 Decoupled inverse on layer params

$$
\gamma_i = \sqrt{\sigma_{Z_i|\cdot}^2 / \sigma_{Z_i}^2}
$$
$$
\mu_{W_{ji}} \leftarrow \gamma_i \mu_{W_{ji}},\quad
\sigma_{W_{ji}}^2 \leftarrow \gamma_i^2 \sigma_{W_{ji}}^2,\quad
\sigma_{B_i}^2 \leftarrow \gamma_i^2 \sigma_{B_i}^2
$$
$$
\tilde\mu_{Z_i} = \gamma_i (\mu_{Z_i} - \mu_{B_i}) + \mu_{B_i},\quad
\Delta\mu_{Z_i} = \mu_{Z_i|\cdot} - \tilde\mu_{Z_i}
$$
$$
\mu_{B_i} \leftarrow \mu_{B_i} + \Delta\mu_{Z_i}
$$

### 2.6 Outer loop

```
for batch in dataloader:                 # one epoch
    ma, Sa = preprocess(x)
    for layer in net:
        if isinstance(layer, LearnableLayer):
            mz, Sz = layer.forward(ma, Sa)            # uncalibrated forward
            calibrate_layer(layer, mz, Sz, sigma_m, sigma_z)
            mz, Sz = layer.forward(ma, Sa)            # re-forward with new params
        else:
            mz, Sz = layer.forward(ma, Sa)            # passthrough (ReLU, etc.)
        ma, Sa = mz, Sz
```

S target is hit exactly per batch (analytical projection). S2 target is
only approached asymptotically across the full dataset because the S2 RTS
update is a linearization of a quadratic observation.

---

## 3. Design decisions (open for review)

These are choices I'll make for V1 unless you say otherwise:

**D1. Aggregation across the batch.** Forward pass gives per-sample
$(\mu_{Z_i}, \sigma_{Z_i}^2)$ of shape $(B, A)$. Two interpretations:

- **(a)** Per-sample S/S2 projection → $(B, A)$ corrected moments → batch-mean
  for the inverse step (single $\gamma_i$, $\Delta\mu_{Z_i}$ per output unit).
- **(b)** Batch-mean first → single $(\mu_{Z_i}, \sigma_{Z_i}^2)$ per output
  unit → S/S2 projection on those scalars → inverse on those scalars.

**Default: (b)** (user decision, 2026-04-23). Per-sample projection is too
expensive; batch-mean-first is the scalar-vector form the §2 formulas are
already written in. Mean over batch is taken on the raw $(\mu_{Z_i}, \sigma_{Z_i}^2)$
entering §2.3; everything after that is per-unit scalars of shape $(A,)$.

**D2. Output layer target — regression.** Same $(\sigma_M, \sigma_Z)$ as
hidden layers. Symmetric, simple.

**D3. Output layer target — classification (Remax head).** User wants
the post-Remax distribution to be uniform $1/C$ at the first batch.
Options:

- **(a)** Calibrate the Linear *before* Remax to a target derived from
  inverting Remax at the uniform output (closed-form for MixtureReLU? probably
  not; would need a numerical / Monte Carlo back-derivation).
- **(b)** Skip the Remax-head calibration in V1; apply standard target on the
  pre-Remax Linear. Verify post-Remax distribution empirically.

**Default: (b)** for V1 — get the MNIST-MLP result first, then revisit.

**D4. Bias prior $\alpha$.** Use whatever the layer's existing init produces.
Don't sweep $\alpha$ in V1.

**D5. Numerical guards.**
- $\sigma_{Z_i}^2 < \epsilon$ → skip the inverse update for unit $i$ (γ undefined).
- $\sigma_S^2, \sigma_{S2}^2 < \epsilon$ → skip the S/S2 update for that batch.

**D6. ReLU passthrough.** ReLU has no learnable params; the calibration
walks past it without modification. The next Linear sees post-ReLU
$(\mu_A, \sigma_A^2)$ as input.

---

## 4. File layout

```
experiments/inference_init/
    PLAN.md                  ← this file
    mnist_mlp_sweep.py       ← reproduces thesis Figure 4 heatmap
    figures/                 ← sweep output (PDF/PNG)
    results/                 ← run logs / .json metrics
triton_tagi/
    inference_init.py        ← the algorithm (lives in the library; clean
                                public API: `inference_init(net, loader,
                                sigma_m, sigma_z)`)
tests/
    unit/test_inference_init.py    ← per-step correctness:
                                     - S projection lands on target
                                     - S2 RTS Jacobian sanity
                                     - decoupled inverse round-trip
    validation/test_ibi_mnist.py   ← end-to-end: 7-layer MLP MNIST,
                                     (σ_M=0.5, σ_Z=0.5), σ_V=0.05,
                                     L=5 → ≥97% (slow, marked @pytest.mark.slow)
```

---

## 5. Phasing

### Phase 1 — Linear + ReLU MLP (this PLAN's V1)

- Implement `inference_init.py` with `Linear` calibration only.
- Pass-through for `ReLU`, `Flatten`, `Remax` (the latter just for completeness;
  per D3, no special treatment in V1).
- Sweep MNIST MLP at L ∈ {1, 3, 5, 7} × $(\sigma_M, \sigma_Z) \in \{0.5, 1.0\}^2$
  × $\sigma_V \in \{0.01, 0.05\}$. Reproduce Figure 4.
- Done when the heatmap qualitatively matches the thesis.

### Phase 2 — Conv2D

- Open theory question: what is "width" $A$ for Conv2D?
  - Options: $C_{\text{out}}$ (per-channel), $C_{\text{out}} \cdot H \cdot W$
    (full feature map), or per-spatial-position with $A = C_{\text{out}}$.
  - Need to decide which produces stable training. Thesis flags this as
    "remaining direction".
- Validate on MNIST CNN, then CIFAR-10 CNN.

### Phase 3 — ResNet-18

- ResBlock has additive shortcut + two Conv2D + BN + ReLU. Two strategies:
  - **(a) Per-conv calibration** — treat each Conv2D atomically; ignore the
    residual structure. Cheapest.
  - **(b) Per-block calibration** — calibrate input → block-output as a unit.
    Theoretically cleaner but the inverse problem is no longer per-layer.
- Validate against the current 89% CIFAR-10 baseline.

---

## 6. Out of scope for V1

- Conv2D / BN / ResBlock / MultiheadAttentionV2 calibration (Phases 2/3).
- Inverse-Remax target derivation (D3 alternative).
- Universal $(\sigma_M, \sigma_Z)$ search across architectures.
- Bias-prior $\alpha$ sweep.
- Multi-epoch calibration (paper says one epoch; we follow that).

---

## 7. Open empirical questions (post-V1, defer)

- Does a universal $(\sigma_M, \sigma_Z)$ exist across architectures?
- Does multi-epoch calibration help, or hurt (over-fitting the prior to the
  calibration set)?
- For deep networks under high $\sigma_V$, is the ReLU-clipping bottleneck
  fundamental or fixable by clipping $\sigma_M$ adaptively per layer?
- Does IBI compose with later regularization (Contribution 3 §regularization,
  not yet read)?
