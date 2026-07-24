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

from .layers.avgpool2d import AvgPool2D as _AvgPool2DLayer
from .layers.batchnorm2d import BatchNorm2D as _BatchNorm2DLayer
from .layers.conv2d import Conv2D as _Conv2DLayer
from .layers.flatten import Flatten as _FlattenLayer
from .layers.linear import Linear as _LinearLayer
from .layers.maxpool2d import MaxPool2D as _MaxPool2DLayer
from .layers.relu import bayesian_relu
from .layers.remax import triton_remax
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
        self.topo: list[GaussianTensor] = []  # backward order (filled by build_topo)

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

    # --- operator sugar: element-wise add / multiply build graph nodes ---
    def __add__(self, other: "GaussianTensor") -> "GaussianTensor":
        return add(self, other)

    def __mul__(self, other: "GaussianTensor") -> "GaussianTensor":
        return mul(self, other)

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

    def build_topo(self) -> list["GaussianTensor"]:
        """Build the backward execution order (children before parents).

        Same construction as PyTorch/micrograd autograd: a depth-first
        post-order over the graph (parents before children), then reversed so
        the sweep visits each node only after all of its children. The result
        is cached on ``self.topo`` for inspection / :meth:`print_graph`.
        """
        post, seen = [], set()

        def dfs(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for p in node.parents:
                dfs(p)
            post.append(node)  # post-order: parents before children

        dfs(self)
        self.topo = list(reversed(post))  # backward order: children before parents
        return self.topo

    def backward(self, cap_factor: float = 1.0) -> None:
        """Reverse-topological sweep: chain the layer backwards, apply capped
        parameter updates. Innovations are freed afterwards (per-forward graph)."""
        for node in self.build_topo():  # children before parents
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

    # ==================================================================
    #  Debugging: visualise the backward graph
    # ==================================================================
    def _kind(self) -> str:
        """Short label for this node's role in the graph."""
        if isinstance(self, Parameter):
            return "Parameter"
        if self._op is None:
            return "input"
        if isinstance(self._op, Activation):
            return f"act:{self._op.name}"
        return type(self._op).__name__  # Add / Mul / Linear

    def render_graph(self, show_moments: bool = False, compact: bool = False) -> str:
        """Return an ASCII rendering of the graph rooted at this node.

        ``compact=False`` (default) draws a nested tree read **top → bottom =
        the backward direction**: the output (this node) is the root, and its
        parents (the nodes it pushes innovations to) hang below it, down to the
        :class:`Parameter` / ``input`` leaves. Nodes reached by more than one
        path (e.g. a shared weight) are expanded once and later shown as
        ``↩ (shared)``.

        ``compact=True`` prints a flat numbered list in backward-execution order
        (children before parents), each node referencing its parents by index —
        far more readable for deep chains (e.g. ResNet) than the nested tree.
        """
        if compact:
            return self._render_compact(show_moments)

        seen: set[int] = set()
        lines: list[str] = []

        def label(node: "GaussianTensor") -> str:
            s = f"{node.name} [{node._kind()}] {tuple(node.shape)}"
            if show_moments:
                s += f"  mu~{_fmt(node.mu)} var~{_fmt(node.var)}"
                if node.d_mu is not None:
                    s += f"  dμ~{_fmt(node.d_mu)} dσ~{_fmt(node.d_var)}"
            return s

        def walk(node: "GaussianTensor", prefix: str, last: bool) -> None:
            conn = "└─ " if last else "├─ "
            dup = id(node) in seen
            lines.append(prefix + conn + label(node) + ("  ↩ (shared)" if dup else ""))
            if dup:
                return
            seen.add(id(node))
            child_prefix = prefix + ("   " if last else "│  ")
            kids = node.parents
            for i, p in enumerate(kids):
                walk(p, child_prefix, i == len(kids) - 1)

        seen.add(id(self))
        lines.append(label(self))
        for i, p in enumerate(self.parents):
            walk(p, "", i == len(self.parents) - 1)

        n = len(self.build_topo())
        header = f"backward graph — {n} nodes (top → bottom = backward flow)"
        return header + "\n" + "─" * len(header) + "\n" + "\n".join(lines)

    def _render_compact(self, show_moments: bool) -> str:
        """Flat numbered list in backward order; parents shown by index."""
        topo = self.build_topo()
        idx = {id(n): i for i, n in enumerate(topo)}

        names = [n.name for n in topo]
        kinds = [n._kind() for n in topo]
        shapes = [str(tuple(n.shape)) for n in topo]
        name_w = min(max((len(s) for s in names), default=4), 24)
        kind_w = max((len(s) for s in kinds), default=4)
        shape_w = min(max((len(s) for s in shapes), default=4), 18)
        idx_w = len(str(len(topo) - 1))

        lines = []
        for i, node in enumerate(topo):
            parents = ", ".join(f"#{idx[id(p)]}" for p in node.parents)
            arrow = f"  ← {parents}" if parents else ""
            s = (
                f"[{i:>{idx_w}}] {names[i]:<{name_w}} "
                f"{kinds[i]:<{kind_w}} {shapes[i]:<{shape_w}}"
            )
            if show_moments:
                s += f"  mu~{_fmt(node.mu)} var~{_fmt(node.var)}"
            lines.append(s + arrow)

        header = (
            f"backward graph — {len(topo)} nodes "
            f"(compact; backward order, #child ← #parent)"
        )
        return header + "\n" + "─" * len(header) + "\n" + "\n".join(lines)

    def print_graph(self, show_moments: bool = False, compact: bool = False) -> None:
        """Print :meth:`render_graph` (handy in a debugger / notebook)."""
        print(self.render_graph(show_moments=show_moments, compact=compact))


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

    def assign_names(self, prefix: str = "") -> "Module":
        """Give every sub-module (and its ``W``/``b`` parameters) a hierarchical
        display name from its attribute path — like torch's ``named_modules`` —
        so the graph renders unambiguously (e.g. ``b2a.conv1`` instead of a bare
        ``conv2d``). Purely cosmetic: it only sets ``.name`` labels and never
        touches the graph wiring, which is by object identity. Call once after
        constructing the network, before building the graph.
        """
        for attr, m in self.__dict__.get("_modules", {}).items():
            m.name = f"{prefix}{attr}"
            for pname in ("W", "b"):
                p = getattr(m, pname, None)
                if isinstance(p, Parameter):
                    p.name = f"{m.name}.{pname}"
            m.assign_names(prefix=f"{m.name}.")
        return self

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
#  Concat:  join two tensors along a dim (U-Net / DenseNet skip connections).
#           Deterministic rearrangement (Jacobian = identity), so the backward
#           just splits the innovation back to each input.
# ----------------------------------------------------------------------


class Concat(Operation):
    def __init__(self, dim: int = 1):
        self.dim = dim

    def forward(self, a: GaussianTensor, b: GaussianTensor) -> GaussianTensor:
        mu = torch.cat([a.mu, b.mu], dim=self.dim)
        var = torch.cat([a.var, b.var], dim=self.dim)
        out = GaussianTensor(mu, var, parents=(a, b), op=self, name="concat")
        out._ctx["split"] = a.mu.shape[self.dim]  # size of the first operand
        return out

    def backward(self, node: GaussianTensor) -> None:
        a, b = node.parents
        s, d = node._ctx["split"], self.dim
        idx_a = [slice(None)] * node.d_mu.dim()
        idx_b = [slice(None)] * node.d_mu.dim()
        idx_a[d] = slice(0, s)
        idx_b[d] = slice(s, None)
        a._accumulate(node.d_mu[tuple(idx_a)], node.d_var[tuple(idx_a)])
        b._accumulate(node.d_mu[tuple(idx_b)], node.d_var[tuple(idx_b)])


def concat(a: GaussianTensor, b: GaussianTensor, dim: int = 1) -> GaussianTensor:
    """Concatenate two Gaussian tensors along ``dim`` (default: channels)."""
    return Concat(dim).forward(a, b)


# ----------------------------------------------------------------------
#  Upsample:  nearest-neighbour spatial upsample by an integer factor.
#           Each input pixel is copied to a k×k output block (Jacobian = 1),
#           so the backward sums each block's innovation back to its pixel
#           (the exact adjoint of the copy).
# ----------------------------------------------------------------------


class Upsample(Operation):
    def __init__(self, scale: int = 2):
        self.scale = scale

    def forward(self, x: GaussianTensor) -> GaussianTensor:
        k = self.scale
        mu = x.mu.repeat_interleave(k, dim=2).repeat_interleave(k, dim=3)
        var = x.var.repeat_interleave(k, dim=2).repeat_interleave(k, dim=3)
        out = GaussianTensor(mu, var, parents=(x,), op=self, name="upsample")
        out._ctx["shape"] = x.shape
        return out

    def backward(self, node: GaussianTensor) -> None:
        (x,) = node.parents
        k = self.scale
        N, C, H, W = node._ctx["shape"]
        dm = node.d_mu.reshape(N, C, H, k, W, k).sum(dim=(3, 5))
        dv = node.d_var.reshape(N, C, H, k, W, k).sum(dim=(3, 5))
        x._accumulate(dm, dv)


def upsample(x: GaussianTensor, scale: int = 2) -> GaussianTensor:
    """Nearest-neighbour spatial upsample by an integer ``scale`` factor."""
    return Upsample(scale).forward(x)


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


def _tanh_moments(mz, Sz):
    """First-order (linearized) Gaussian moments of tanh."""
    mu = torch.tanh(mz)
    jcb = 1.0 - mu * mu                    # tanh'(z) = 1 - tanh(z)^2
    var = (jcb * jcb * Sz).clamp_min(0.0)
    return mu, var, jcb


def _sigmoid_moments(mz, Sz):
    """First-order (linearized) Gaussian moments of the logistic sigmoid."""
    mu = torch.sigmoid(mz)
    jcb = mu * (1.0 - mu)                  # sigmoid'(z) = s(z)(1 - s(z))
    var = (jcb * jcb * Sz).clamp_min(0.0)
    return mu, var, jcb


def tanh(z: GaussianTensor) -> GaussianTensor:
    """Bayesian tanh activation (first-order linearized moments)."""
    return Activation(_tanh_moments, "tanh").forward(z)


def sigmoid(z: GaussianTensor) -> GaussianTensor:
    """Bayesian logistic-sigmoid activation (first-order linearized moments)."""
    return Activation(_sigmoid_moments, "sigmoid").forward(z)


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


class Tanh(Module):
    """Torch-style tanh activation module (wraps :func:`tanh`)."""

    def forward(self, z: GaussianTensor) -> GaussianTensor:
        return tanh(z)

    def __repr__(self):
        return "Tanh()"


class Sigmoid(Module):
    """Torch-style sigmoid activation module (wraps :func:`sigmoid`)."""

    def forward(self, z: GaussianTensor) -> GaussianTensor:
        return sigmoid(z)

    def __repr__(self):
        return "Sigmoid()"


# ----------------------------------------------------------------------
#  Linear:  reuses triton_tagi.layers.Linear forward + backward as-is.
# ----------------------------------------------------------------------


class Linear(Module):
    """Bayesian fully-connected layer as an autocov operation.

    A thin wrapper that **reuses** :class:`triton_tagi.layers.Linear` — its
    ``forward`` (matmul + fused variance kernel) and its ``backward`` (fused
    backward-delta for the input plus the stored weight/bias deltas) are called
    directly, so there is no duplicated linear math.

    The layer's Gaussian weight/bias are exposed as autocov :class:`Parameter`
    leaves that **alias the layer's own tensors** (same objects, no copy), so the
    backward sweep applies the capped cuTAGI update in place and the wrapped layer
    sees the updated weights on the next forward.

    To set parameters after construction, mutate in place
    (``lin.W.mu.copy_(...)``) so the alias is preserved. Like the underlying layer
    (and autograd in general), the per-forward activation cache lives on the layer
    instance, so call ``observe``/backward before reusing the same ``Linear``
    instance elsewhere in the graph.
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

        # Reuse the library layer for forward, backward, and initialisation.
        self._layer = _LinearLayer(
            in_features, out_features, device=device, init_method=init_method,
            gain_w=gain_w, gain_b=gain_b, bias=bias, generator=rng,
        )
        # Expose (and alias) the layer's parameters so the sweep updates them.
        self.W = Parameter(self._layer.mw, self._layer.Sw, name=f"{name}.W")
        self.W.mu, self.W.var = self._layer.mw, self._layer.Sw  # guarantee aliasing
        if bias:
            self.b = Parameter(self._layer.mb, self._layer.Sb, name=f"{name}.b")
            self.b.mu, self.b.var = self._layer.mb, self._layer.Sb
        else:
            self.b = None

    def forward(self, a: GaussianTensor) -> GaussianTensor:
        mz, Sz = self._layer.forward(a.mu, a.var)  # reuse existing forward
        parents = (a, self.W, self.b) if self.has_bias else (a, self.W)
        return GaussianTensor(mz, Sz, parents=parents, op=self, name=self.name)

    def backward(self, node: GaussianTensor) -> None:
        a = node.parents[0]
        # Restore this node's cached input on the wrapped layer. The layer caches
        # ma_in on the instance during forward; when the SAME Linear is reused in
        # one graph (weight sharing / RNN unroll) that cache holds only the last
        # forward, so we re-inject the correct per-node input from the graph edge.
        self._layer.ma_in = a.mu.reshape(-1, self.in_features)
        self._layer._input_shape = a.mu.shape
        # Reuse existing backward: returns input deltas, stores the param deltas.
        d_ma, d_Sa = self._layer.backward(node.d_mu, node.d_var)
        a._accumulate(d_ma, d_Sa)
        # Route the layer's stored parameter deltas onto the aliased Parameters;
        # the sweep then applies the capped update in place (layer tensors too).
        # Reused instances accumulate across every use → backprop-through-time.
        self.W._accumulate(self._layer.delta_mw, self._layer.delta_Sw)
        if self.has_bias:
            self.b._accumulate(self._layer.delta_mb, self._layer.delta_Sb)

    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features}, bias={self.has_bias})"


# ----------------------------------------------------------------------
#  Conv2D:  reuses triton_tagi.layers.Conv2D forward + backward as-is.
# ----------------------------------------------------------------------


class Conv2D(Module):
    """Bayesian 2-D convolution as an autocov operation.

    A thin wrapper that **reuses** :class:`triton_tagi.layers.Conv2D` — its
    ``forward`` (im2col + fused variance kernel) and its ``backward`` (fused
    backward-delta for the input plus the stored weight/bias deltas) are called
    directly, so there is no duplicated conv math.

    The layer's Gaussian weight/bias are exposed as autocov :class:`Parameter`
    leaves that **alias the layer's own tensors** (same objects, no copy). The
    backward sweep therefore applies the capped cuTAGI update in place, and the
    wrapped layer sees the updated weights on the next forward.

    Note: like the underlying layer (and autograd in general), the per-forward
    activation cache lives on the layer instance, so call ``observe``/backward
    before reusing the same ``Conv2D`` instance again in the graph. Use separate
    instances if you need two conv sites in one graph.
    """

    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        padding_type: int = 1,
        device: str = "cpu",
        init_method: str = "He",
        gain_w: float = 1.0,
        gain_b: float = 1.0,
        name: str = "conv2d",
    ) -> None:
        super().__init__()
        self.name = name
        # Reuse the library layer for forward, backward, and initialisation.
        self._layer = _Conv2DLayer(
            C_in, C_out, kernel_size,
            stride=stride, padding=padding, padding_type=padding_type,
            device=device, init_method=init_method, gain_w=gain_w, gain_b=gain_b,
        )
        # Expose (and alias) the layer's parameters so the sweep updates them.
        self.W = Parameter(self._layer.mw, self._layer.Sw, name=f"{name}.W")
        self.b = Parameter(self._layer.mb, self._layer.Sb, name=f"{name}.b")
        self.W.mu, self.W.var = self._layer.mw, self._layer.Sw  # guarantee aliasing
        self.b.mu, self.b.var = self._layer.mb, self._layer.Sb

    def forward(self, a: GaussianTensor) -> GaussianTensor:
        mz, Sz = self._layer.forward(a.mu, a.var)  # reuse existing forward
        return GaussianTensor(mz, Sz, parents=(a, self.W, self.b), op=self, name=self.name)

    def backward(self, node: GaussianTensor) -> None:
        a = node.parents[0]
        # Reuse existing backward: returns input deltas, stores the param deltas.
        d_ma, d_Sa = self._layer.backward(node.d_mu, node.d_var)
        a._accumulate(d_ma, d_Sa)
        # Route the layer's stored parameter deltas onto the aliased Parameters;
        # the sweep then applies the capped update in place (layer tensors too).
        self.W._accumulate(self._layer.delta_mw, self._layer.delta_Sw)
        self.b._accumulate(self._layer.delta_mb, self._layer.delta_Sb)

    def __repr__(self):
        L = self._layer
        return f"Conv2D({L.C_in}, {L.C_out}, kernel={L.kH}, stride={L.stride}, pad={L.padding})"


# ----------------------------------------------------------------------
#  BatchNorm2D:  reuses triton_tagi.layers.BatchNorm2D forward + backward.
# ----------------------------------------------------------------------


class BatchNorm2D(Module):
    """Bayesian channel-wise BatchNorm2D as an autocov operation.

    A thin wrapper that **reuses** :class:`triton_tagi.layers.BatchNorm2D` — its
    ``forward`` (normalise with batch/running stats + Gaussian affine) and its
    ``backward`` (un-normalise + stored γ/β deltas) are called directly.

    The learnable scale ``γ`` and shift ``β`` are exposed as autocov
    :class:`Parameter` leaves (``W`` and ``b``) that **alias the layer's tensors**,
    so the backward sweep applies the capped update in place. Because BatchNorm's
    data-dependent ``preserve_var`` init reassigns ``γ`` on the first forward, the
    aliases are re-synced after every forward.

    ``train()`` / ``eval()`` switch the wrapped layer between batch and running
    statistics (and recurse like any :class:`Module`).
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
        device: str = "cpu",
        gain_w: float = 1.0,
        gain_b: float = 1.0,
        preserve_var: bool = True,
        name: str = "batchnorm2d",
    ) -> None:
        super().__init__()
        self.name = name
        self._layer = _BatchNorm2DLayer(
            num_features, momentum=momentum, eps=eps, device=device,
            gain_w=gain_w, gain_b=gain_b, preserve_var=preserve_var,
        )
        self.W = Parameter(self._layer.mw, self._layer.Sw, name=f"{name}.W")  # γ
        self.b = Parameter(self._layer.mb, self._layer.Sb, name=f"{name}.b")  # β
        self._sync_params()

    def _sync_params(self) -> None:
        """Re-point the Parameter aliases at the layer's live tensors (γ may be
        reassigned by the layer's data-dependent ``preserve_var`` init)."""
        self.W.mu, self.W.var = self._layer.mw, self._layer.Sw
        self.b.mu, self.b.var = self._layer.mb, self._layer.Sb

    def forward(self, a: GaussianTensor) -> GaussianTensor:
        ma, Sa = self._layer.forward(a.mu, a.var)  # reuse existing forward
        self._sync_params()  # keep aliases current after any γ reassignment
        return GaussianTensor(ma, Sa, parents=(a, self.W, self.b), op=self, name=self.name)

    def backward(self, node: GaussianTensor) -> None:
        a = node.parents[0]
        # Reuse existing backward: returns input deltas, stores γ/β deltas.
        d_mz, d_Sz = self._layer.backward(node.d_mu, node.d_var)
        a._accumulate(d_mz, d_Sz)
        self.W._accumulate(self._layer.delta_mw, self._layer.delta_Sw)
        self.b._accumulate(self._layer.delta_mb, self._layer.delta_Sb)

    def train(self, mode: bool = True) -> "Module":
        self._layer.train() if mode else self._layer.eval()
        return super().train(mode)

    def eval(self) -> "Module":
        return self.train(False)

    def __repr__(self):
        L = self._layer
        return f"BatchNorm2D(num_features={L.num_features}, momentum={L.momentum}, eps={L.eps})"


# ----------------------------------------------------------------------
#  Parameter-free ops: AvgPool2D / MaxPool2D / Flatten
#
#  Each wraps a single-input, single-output triton_tagi layer and reuses its
#  forward + backward verbatim. No parameters, so the backward just routes the
#  input-space delta onto the one parent.
# ----------------------------------------------------------------------


class _WrappedUnary(Module):
    """Base for param-free autocov ops that wrap a one-in/one-out layer.

    Subclasses set ``self.name`` and ``self._layer`` in ``__init__``. The wrapped
    layer caches per-forward state (pooling indices, input shape) on its instance,
    so reusing one op in a graph (e.g. one pool applied at several U-Net stages)
    would clobber it. Because these layers are parameter-free, the backward simply
    recomputes that cache from the node's own input first — making reuse safe.
    """

    def forward(self, a: GaussianTensor) -> GaussianTensor:
        ma, Sa = self._layer.forward(a.mu, a.var)  # reuse existing forward
        return GaussianTensor(ma, Sa, parents=(a,), op=self, name=self.name)

    def backward(self, node: GaussianTensor) -> None:
        a = node.parents[0]
        # Restore this node's per-forward cache (the op instance may be reused
        # elsewhere in the graph, which would have overwritten it). Safe because
        # the wrapped layer has no learnable parameters or running state.
        self._layer.forward(a.mu, a.var)
        d_ma, d_Sa = self._layer.backward(node.d_mu, node.d_var)  # reuse existing backward
        a._accumulate(d_ma, d_Sa)

    def __repr__(self):
        return repr(self._layer)


class AvgPool2D(_WrappedUnary):
    """Bayesian average pooling as an autocov op (wraps triton_tagi.layers.AvgPool2D)."""

    def __init__(
        self,
        kernel_size: int,
        spatial_correlation: bool = False,
        name: str = "avgpool2d",
    ) -> None:
        super().__init__()
        self.name = name
        self._layer = _AvgPool2DLayer(kernel_size, spatial_correlation=spatial_correlation)


class MaxPool2D(_WrappedUnary):
    """Bayesian max pooling as an autocov op (wraps triton_tagi.layers.MaxPool2D)."""

    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        name: str = "maxpool2d",
    ) -> None:
        super().__init__()
        self.name = name
        self._layer = _MaxPool2DLayer(kernel_size, stride=stride, padding=padding)


class Flatten(_WrappedUnary):
    """Flatten spatial dims (N,C,H,W)→(N,C·H·W) as an autocov op."""

    def __init__(self, name: str = "flatten") -> None:
        super().__init__()
        self.name = name
        self._layer = _FlattenLayer()
