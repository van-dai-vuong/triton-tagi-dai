"""
autocov — automatic covariance / "autograd for TAGI".
=====================================================

A tiny reverse-mode engine for Tractable Approximate Gaussian Inference.

Every value in the graph is a :class:`GaussianTensor` ``N(mu, diag(var))``.
Each operation records its inputs (``parents``) and the op that produced it
(``_op``, the analogue of autograd's ``grad_fn``). Calling
:meth:`GaussianTensor.observe` conditions the output on a noisy measurement and
then automatically sweeps the innovation backwards through the whole graph,
updating every :class:`Parameter` in place — no manual per-layer backward.

Innovation convention — the repo's normalized innovation
--------------------------------------------------------
This engine carries the **same normalized innovation** the rest of
``triton_tagi`` already uses (``update.observation.compute_innovation``):

    delta_mu  = (y - mu) / (var + var_v)          # = cov(Y,Z)^{-1}-scaled resid.
    delta_var = -1        / (var + var_v)

Every hidden node accumulates ``(delta_mu, delta_var)`` in exactly this space,
so each operation's backward is **identical to the corresponding layer's
backward** in ``triton_tagi.layers`` — no per-node variance rescaling:

    Add   S = X + Y :  delta_X = delta_S                 (identity)
    Mul   Z = X * Y :  delta_mu_X = mu_Y  * delta_mu_Z   (element-wise)
                       delta_var_X = mu_Y^2 * delta_var_Z
    Act   A = f(Z)  :  delta_mu_Z = jcb  * delta_mu_A     (jcb from the layer)
                       delta_var_Z = jcb^2 * delta_var_A
    Lin   Z = a·W+b :  (delta_a) = fused_backward_delta(delta_z, mw)   [input]
                       delta_mw = Sw · (a^T @ delta_mu_z)              [weights]
                       delta_Sw = Sw^2 · ((a^2)^T @ delta_var_z)
                       delta_mb = Sb · Σ delta_mu_z                    [bias]

Parameters accumulate their *actual* capped-update deltas (the ``Sw·grad`` /
``Sb·grad`` products above) and apply them with the library's cuTAGI-style
:func:`~triton_tagi.update.parameters.update_parameters`.

The mean / variance / Jacobian of activations and the Linear layer reuse the
existing kernels (``bayesian_relu``, ``triton_remax``,
``triton_fused_var_forward`` / ``triton_fused_backward_delta`` /
``triton_fused_weight_grad``), so numerics match the rest of ``triton_tagi``.

Example (Torch-style Module)
----------------------------
    from triton_tagi.autocov import Module, Linear, ReLU, tensor

    class MLP(Module):
        def __init__(self):
            super().__init__()
            self.fc1 = Linear(1, 16)
            self.act = ReLU()
            self.fc2 = Linear(16, 1)
        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    net = MLP()
    for x, y in data:                          # x, y : python floats or tensors
        out = net(tensor([[x]], var=0.0))
        out.observe([[y]], var_v=0.1)          # updates all net params in place

Example (functional)
--------------------
    from triton_tagi.autocov import Linear, relu, tensor

    lin1, lin2 = Linear(1, 16), Linear(16, 1)
    for x, y in data:
        z = lin2(relu(lin1(tensor([[x]]))))
        z.observe([[y]], var_v=0.1)
"""

from __future__ import annotations

import torch
from torch import Tensor

from .kernels.common import (
    triton_fused_backward_delta,
    triton_fused_var_forward,
    triton_fused_weight_grad,
)
from .layers.relu import bayesian_relu
from .layers.remax import triton_remax
from .param_init import init_weight_bias_linear
from .update.parameters import get_cap_factor, update_parameters

_EPS = 1e-12  # floor for retained POSTERIOR variances (must stay > 0)
_DIV_EPS = 1e-12  # floor for the predictive-variance denominator in observe()

_TRACE = False


def set_trace(enabled: bool = True) -> None:
    """Print each step of the backward sweep (seed, closures, parameter sinks).

    Zero overhead when disabled.
    """
    global _TRACE
    _TRACE = enabled


def _fmt(v: Tensor) -> str:
    v = torch.as_tensor(v, dtype=torch.float64)
    if v.numel() == 1:
        return f"{float(v.reshape(-1)[0]):+.4f}"
    return f"[{tuple(v.shape)} max|.|={float(v.abs().max()):.4f}]"


# ======================================================================
#  GaussianTensor — the graph node
# ======================================================================


class GaussianTensor:
    """A Gaussian random tensor ``N(mu, diag(var))`` with autograd-style graph
    bookkeeping.

    Attributes
    ----------
    mu, var    : the prior moments (``var`` is the diagonal of the covariance).
    parents    : the GaussianTensor inputs of the op that produced this node.
    _op        : the :class:`Operation` that produced this node (``grad_fn``).
    _ctx       : per-forward cache the op stashes for its backward.
    d_mu, d_var: accumulated **normalized innovations** for hidden nodes, or the
                 accumulated **actual update deltas** for :class:`Parameter`
                 leaves. The analogue of ``.grad``; ``None`` until something
                 flows in.
    post_mu, post_var : filled after :meth:`backward` when ``retain()`` was set
                 (the true posterior ``mu + var·delta_mu`` / ``var + var^2·delta_var``).
    """

    def __init__(self, mu, var, parents=(), op=None, name=""):
        self.mu = torch.as_tensor(mu, dtype=torch.get_default_dtype())
        self.var = torch.as_tensor(var, dtype=torch.get_default_dtype())
        if self.var.shape != self.mu.shape:
            self.var = self.var.expand_as(self.mu).clone()
        self.parents = tuple(parents)
        self._op = op
        self._ctx: dict = {}
        self.name = name
        self.d_mu: Tensor | None = None
        self.d_var: Tensor | None = None
        self._retain = False
        self.post_mu: Tensor | None = None
        self.post_var: Tensor | None = None

    # ------------------------------------------------------------------ utils
    @property
    def shape(self):
        return self.mu.shape

    def std(self) -> Tensor:
        return torch.sqrt(self.var)

    def named(self, name: str) -> "GaussianTensor":
        """Rename this node (chainable) for readable traces."""
        self.name = name
        return self

    def retain(self) -> "GaussianTensor":
        """Ask :meth:`backward` to store this node's posterior in
        ``post_mu`` / ``post_var`` (chainable)."""
        self._retain = True
        return self

    def __repr__(self):
        return (
            f"GaussianTensor(name={self.name!r}, shape={tuple(self.shape)}, "
            f"mu~{self.mu.reshape(-1)[:3].tolist()}, "
            f"var~{self.var.reshape(-1)[:3].tolist()})"
        )

    def _accumulate(self, d_mu: Tensor, d_var: Tensor) -> None:
        if _TRACE:
            print(f"      OUT -> {self.name}._accumulate(d_mu={_fmt(d_mu)}, d_var={_fmt(d_var)})")
        if self.d_mu is None:
            self.d_mu = torch.zeros_like(self.mu)
            self.d_var = torch.zeros_like(self.var)
        self.d_mu = self.d_mu + d_mu
        self.d_var = self.d_var + d_var

    # ==================================================================
    #  observe(): output-layer normalized innovation + automatic backward.
    # ==================================================================
    def observe(self, y, var_v: float = 0.0, cap_factor: float | None = None) -> None:
        """Condition the graph on ``y = self + v``, ``v ~ N(0, var_v)``, then
        propagate the update back to every hidden state and parameter.
        Parameters are updated IN PLACE (capped, cuTAGI-style).

        The seed is the repo's normalized innovation
        (``update.observation.compute_innovation``)::

            delta_mu  = (y - mu) / (var + var_v)
            delta_var = -1       / (var + var_v)
        """
        y = torch.as_tensor(y, dtype=self.mu.dtype).reshape(self.mu.shape)
        if _TRACE:
            print(f"[SEED]    observe(y={_fmt(y)}, var_v={var_v}) on {self.name!r}")
        var_y = (self.var + var_v).clamp_min(_DIV_EPS)  # predictive variance
        delta_mu = (y - self.mu) / var_y  # normalized innovation
        delta_var = -1.0 / var_y
        if cap_factor is None:
            batch = int(self.mu.shape[0]) if self.mu.dim() >= 1 else 1
            cap_factor = get_cap_factor(batch)
        self._accumulate(delta_mu, delta_var)
        self.backward(cap_factor)

    def backward(self, cap_factor: float = 1.0) -> None:
        """Reverse-topological sweep: chain the layer backwards, apply capped
        parameter updates. Innovations are freed afterwards (per-forward graph)."""
        topo, seen = [], set()

        def dfs(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for p in node.parents:
                dfs(p)
            topo.append(node)

        dfs(self)
        for node in reversed(topo):  # children before parents
            has_innov = node.d_mu is not None
            if node._retain and not isinstance(node, Parameter):
                # True posterior: hidden nodes carry NORMALIZED innovation, so
                # the posterior change is var·delta_mu / var^2·delta_var.
                dm = node.var * node.d_mu if has_innov else 0.0
                dv = (node.var * node.var) * node.d_var if has_innov else 0.0
                node.post_mu = node.mu + dm
                node.post_var = torch.clamp(node.var + dv, min=_EPS)
            if not has_innov:
                continue
            if node._op is not None:
                if _TRACE:
                    print(
                        f"[CLOSURE] {node.name}._backward  IN: "
                        f"d_mu={_fmt(node.d_mu)}, d_var={_fmt(node.d_var)}"
                    )
                node._op.backward(node)
            if isinstance(node, Parameter):
                node._apply_update(cap_factor)
                if node._retain:
                    node.post_mu, node.post_var = node.mu, node.var
            # free innovations (graph is per-forward, like autograd)
            node.d_mu = node.d_var = None


class Parameter(GaussianTensor):
    """A leaf Gaussian tensor whose ``(mu, var)`` persist and get updated with
    the library's capped cuTAGI update. Its ``d_mu`` / ``d_var`` hold the actual
    ``Sw·grad`` / ``Sw^2·grad`` deltas accumulated by the consuming op."""

    def __init__(self, mu, var, name="param"):
        super().__init__(mu, var, parents=(), op=None, name=name)

    def _apply_update(self, cap_factor: float) -> None:
        if _TRACE:
            print(f"[SINK]    {self.name}: capped update (cap_factor={cap_factor})")
        # In-place, capped, matching triton_tagi's parameter update.
        update_parameters(self.mu, self.var, self.d_mu, self.d_var, cap_factor)


def tensor(mu, var=0.0, name="x") -> GaussianTensor:
    """Wrap data (e.g. deterministic covariates: ``var=0``) as a graph leaf."""
    return GaussianTensor(mu, var, name=name)


# ======================================================================
#  Module — Torch-style container
#
#  Subclass it, create sub-modules / :class:`Parameter` leaves in ``__init__``,
#  and build the graph in ``forward``. Assigning a ``Module`` or ``Parameter``
#  as an attribute auto-registers it, so ``parameters()`` / ``named_parameters()``
#  discover the whole tree — exactly like ``torch.nn.Module``.
#
#      class MLP(Module):
#          def __init__(self):
#              super().__init__()
#              self.fc1 = Linear(1, 16)
#              self.act = ReLU()
#              self.fc2 = Linear(16, 1)
#          def forward(self, x):
#              return self.fc2(self.act(self.fc1(x)))
#
#      net = MLP()
#      net(tensor([[x]])).observe([[y]], var_v=0.1)   # trains net in place
# ======================================================================


class Module:
    """Base class for autocov networks (Torch ``nn.Module`` analogue).

    Training needs no explicit optimizer/backward: build the graph in
    :meth:`forward`, then call ``.observe(...)`` on the output — the reverse
    sweep updates every registered :class:`Parameter` in place.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    # --- attribute registration (auto-tracks Parameters and sub-Modules) ---
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.__dict__.setdefault("_params", {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        object.__setattr__(self, name, value)

    # --- forward / call ---
    def forward(self, *args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError("Module subclasses must implement forward()")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # --- introspection ---
    def named_parameters(self, prefix: str = ""):
        """Yield ``(name, Parameter)`` for every parameter in the tree."""
        for n, p in self.__dict__.get("_params", {}).items():
            yield f"{prefix}{n}", p
        for mn, m in self.__dict__.get("_modules", {}).items():
            yield from m.named_parameters(prefix=f"{prefix}{mn}.")

    def parameters(self):
        """Yield every :class:`Parameter` in the tree (Torch-style)."""
        for _, p in self.named_parameters():
            yield p

    def modules(self):
        """Yield this module and all sub-modules (self first)."""
        yield self
        for m in self.__dict__.get("_modules", {}).values():
            yield from m.modules()

    @property
    def num_parameters(self) -> int:
        """Total learnable scalars (mean + variance for each parameter tensor)."""
        return sum(2 * p.mu.numel() for p in self.parameters())

    # --- train / eval mode (recurses; no-op for the current op set) ---
    def train(self, mode: bool = True) -> "Module":
        object.__setattr__(self, "training", mode)
        for m in self.__dict__.get("_modules", {}).values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}("]
        for mn, m in self.__dict__.get("_modules", {}).items():
            lines.append(f"  ({mn}): {m!r}")
        for pn, p in self.__dict__.get("_params", {}).items():
            lines.append(f"  ({pn}): Parameter{tuple(p.shape)}")
        lines.append(")")
        return "\n".join(lines) if len(lines) > 2 else f"{type(self).__name__}()"


# ======================================================================
#  Operations — each has forward() and backward()
#
#  forward(*inputs) -> GaussianTensor
#      builds the output node, records parents, and stashes anything the
#      backward needs in ``out._ctx``.
#  backward(node)
#      reads the node's normalized innovation (node.d_mu / node.d_var) and
#      accumulates each parent's contribution — identical to the matching
#      layer backward in ``triton_tagi.layers``.
# ======================================================================


class Operation:
    """Base class. Op instances are stateless w.r.t. a single forward: every
    per-call quantity lives on the produced node (``node.parents`` / ``node._ctx``),
    so one instance may be reused many times in a graph (e.g. weight sharing)."""

    def forward(self, *inputs) -> GaussianTensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def backward(self, node: GaussianTensor) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, *inputs) -> GaussianTensor:
        return self.forward(*inputs)


# ----------------------------------------------------------------------
#  Add:  S = X + Y     (independent → variances add; delta passes through)
# ----------------------------------------------------------------------


class Add(Operation):
    def forward(self, x: GaussianTensor, y: GaussianTensor) -> GaussianTensor:
        if x.shape != y.shape:
            raise ValueError(f"Add expects matching shapes, got {x.shape} and {y.shape}")
        return GaussianTensor(x.mu + y.mu, x.var + y.var, parents=(x, y), op=self, name="add")

    def backward(self, node: GaussianTensor) -> None:
        x, y = node.parents
        # Normalized innovation is identical for both summands (cov = var each).
        x._accumulate(node.d_mu, node.d_var)
        y._accumulate(node.d_mu, node.d_var)


def add(x: GaussianTensor, y: GaussianTensor) -> GaussianTensor:
    """Element-wise sum of two independent Gaussian tensors."""
    return Add().forward(x, y)


# ----------------------------------------------------------------------
#  Mul:  Z = X * Y (element-wise, independent)
#        mu_Z  = mu_X mu_Y
#        var_Z = var_X var_Y + var_X mu_Y^2 + mu_X^2 var_Y
#        backward (normalized):  delta_mu_X = mu_Y·delta_mu_Z,
#                                delta_var_X = mu_Y^2·delta_var_Z
# ----------------------------------------------------------------------


class Mul(Operation):
    def forward(self, x: GaussianTensor, y: GaussianTensor) -> GaussianTensor:
        if x.shape != y.shape:
            raise ValueError(f"Mul expects matching shapes, got {x.shape} and {y.shape}")
        mu = x.mu * y.mu
        var = x.var * y.var + x.var * (y.mu * y.mu) + (x.mu * x.mu) * y.var
        return GaussianTensor(mu, var, parents=(x, y), op=self, name="mul")

    def backward(self, node: GaussianTensor) -> None:
        x, y = node.parents
        x._accumulate(y.mu * node.d_mu, (y.mu * y.mu) * node.d_var)
        y._accumulate(x.mu * node.d_mu, (x.mu * x.mu) * node.d_var)


def mul(x: GaussianTensor, y: GaussianTensor) -> GaussianTensor:
    """Element-wise product of two independent Gaussian tensors."""
    return Mul().forward(x, y)


# ----------------------------------------------------------------------
#  Activation:  A = f(Z), (mu_A, var_A, jcb) from the existing library kernel.
#        backward (normalized): delta_mu_Z = jcb·delta_mu_A,
#                               delta_var_Z = jcb^2·delta_var_A
#        (identical to the layer backward, e.g. triton_tagi.layers.ReLU.)
# ----------------------------------------------------------------------


class Activation(Operation):
    """Generic activation wrapper around a moment function returning
    ``(mu_a, var_a, jcb)`` — e.g. ``bayesian_relu`` or ``triton_remax``."""

    def __init__(self, fn, name: str):
        self.fn = fn
        self.name = name

    def forward(self, z: GaussianTensor) -> GaussianTensor:
        mu_a, var_a, jcb = self.fn(z.mu, z.var)
        out = GaussianTensor(mu_a, var_a, parents=(z,), op=self, name=self.name)
        out._ctx["jcb"] = jcb
        return out

    def backward(self, node: GaussianTensor) -> None:
        (z,) = node.parents
        jcb = node._ctx["jcb"]
        z._accumulate(jcb * node.d_mu, (jcb * jcb) * node.d_var)


def relu(z: GaussianTensor) -> GaussianTensor:
    """Bayesian ReLU activation (exact rectified-Gaussian moments)."""
    return Activation(bayesian_relu, "relu").forward(z)


def remax(z: GaussianTensor) -> GaussianTensor:
    """Remax activation (softmax alternative), row-wise over the last dim."""
    return Activation(triton_remax, "remax").forward(z)


class ReLU(Module):
    """Torch-style ReLU activation module (wraps :func:`relu`)."""

    def forward(self, z: GaussianTensor) -> GaussianTensor:
        return relu(z)

    def __repr__(self):
        return "ReLU()"


class Remax(Module):
    """Torch-style Remax activation module (wraps :func:`remax`)."""

    def forward(self, z: GaussianTensor) -> GaussianTensor:
        return remax(z)

    def __repr__(self):
        return "Remax()"


# ----------------------------------------------------------------------
#  Linear:  Z = a·W + b     (a, W, b Gaussian; W, b are Parameters)
#        Forward moments and the three backward reductions all reuse the
#        library kernels, so this matches triton_tagi.layers.Linear exactly.
# ----------------------------------------------------------------------


class Linear(Module):
    """Bayesian fully-connected layer (Torch-style module).

    Holds its Gaussian weight/bias as :class:`Parameter` leaves — auto-registered
    by :class:`Module`, so ``net.parameters()`` finds them — and defines the op
    :meth:`backward` the graph sweep calls. Forward moments and the three
    backward reductions reuse the library kernels, matching
    ``triton_tagi.layers.Linear`` exactly.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: str = "cpu",
        init_method: str = "He",
        gain_w: float = 1.0,
        gain_b: float = 1.0,
        bias: bool = True,
        rng: "torch.Generator | None" = None,
        name: str = "linear",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.name = name

        mw, Sw, mb, Sb = init_weight_bias_linear(
            in_features, out_features,
            init_method=init_method, gain_w=gain_w, gain_b=gain_b,
            bias=bias, device=device, generator=rng,
        )
        self.W = Parameter(mw, Sw, name=f"{name}.W")
        # A learnable bias registers as a Parameter; otherwise keep a fixed,
        # zero-variance deterministic bias (not registered / not updated).
        if bias:
            self.b = Parameter(mb, Sb, name=f"{name}.b")
        else:
            self.b = GaussianTensor(mb, Sb, name=f"{name}.b")

    def forward(self, a: GaussianTensor) -> GaussianTensor:
        if a.shape[-1] != self.in_features:
            raise ValueError(f"Linear expected last dim {self.in_features}, got {a.shape[-1]}")
        lead = a.shape[:-1]
        ma = a.mu.reshape(-1, self.in_features)
        va = a.var.reshape(-1, self.in_features)

        mu_z = (ma @ self.W.mu + self.b.mu).reshape(*lead, self.out_features)
        Sz = triton_fused_var_forward(ma, va, self.W.mu, self.W.var, self.b.var)
        var_z = Sz.reshape(*lead, self.out_features)

        parents = (a, self.W, self.b) if self.has_bias else (a, self.W)
        return GaussianTensor(mu_z, var_z, parents=parents, op=self, name=self.name)

    def backward(self, node: GaussianTensor) -> None:
        a, W = node.parents[0], node.parents[1]
        b = node.parents[2] if self.has_bias else None
        out_f, in_f = self.out_features, self.in_features

        # Output normalized innovation, flattened to 2D for the kernels.
        dmu_z = node.d_mu.reshape(-1, out_f)
        dvar_z = node.d_var.reshape(-1, out_f)

        # ── Input a: same as triton_tagi.layers.Linear input-delta propagation ──
        d_ma, d_Sa = triton_fused_backward_delta(dmu_z, dvar_z, W.mu)
        a._accumulate(d_ma.reshape(a.mu.shape), d_Sa.reshape(a.var.shape))

        # ── Weight W: Δμ_W = Sw·(a^T @ δμ_z),  ΔS_W = Sw^2·((a^2)^T @ δS_z) ──
        ma = a.mu.reshape(-1, in_f)
        grad_mw, grad_Sw = triton_fused_weight_grad(ma, dmu_z, dvar_z)
        W._accumulate(W.var * grad_mw, (W.var * W.var) * grad_Sw)

        # ── Bias b: Δμ_b = Sb·Σ δμ_z,  ΔS_b = Sb^2·Σ δS_z ──
        if self.has_bias:
            b._accumulate(
                b.var * dmu_z.sum(0, keepdim=True),
                (b.var * b.var) * dvar_z.sum(0, keepdim=True),
            )

    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features}, bias={self.has_bias})"
