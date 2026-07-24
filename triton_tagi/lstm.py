"""
Bayesian LSTM for the :mod:`triton_tagi.autocov` graph engine.

Built entirely from autocov primitives — :class:`~triton_tagi.autocov.Linear`
plus the gate ops ``sigmoid`` / ``tanh`` / ``*`` / ``+`` — so training,
backprop-through-time, and per-parameter TAGI updates all happen automatically
when you call ``observe()`` on the output. There is no bespoke LSTM backward:
the graph the cell builds while unrolling *is* the BPTT graph.

Cell equations (separate input/hidden projections, additive pre-activations)::

    i = sigmoid(Wi x + Ui h)          input gate
    f = sigmoid(Wf x + Uf h)          forget gate
    g = tanh   (Wg x + Ug h)          candidate
    o = sigmoid(Wo x + Uo h)          output gate
    c' = f * c + i * g                cell state
    h' = o * tanh(c')                 hidden state

Weight sharing across timesteps works because the autocov ``Linear`` backward
re-injects each node's own cached input from the graph edge (so reusing one
``Linear`` instance across the unroll accumulates its parameter deltas over all
timesteps — exactly backprop-through-time).

Usage::

    from triton_tagi.layers.lstm import LSTM
    from triton_tagi.autocov import Linear, Module, tensor

    class Net(Module):
        def __init__(self):
            super().__init__()
            self.lstm = LSTM(input_size=1, hidden_size=16)
            self.readout = Linear(16, 1)
        def forward(self, seq):           # seq: list of (B, 1) GaussianTensors
            return self.readout(self.lstm(seq))
"""

from __future__ import annotations

import torch

from .autocov import (
    GaussianTensor,
    Linear,
    Module,
    get_default_device,
    sigmoid,
    tanh,
    tensor,
)


class LSTMCell(Module):
    """A single Bayesian LSTM step, composed of autocov ``Linear`` gates.

    Uses separate input (``W*``, with bias) and hidden (``U*``, no bias)
    projections per gate, summed before the nonlinearity.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rng: "torch.Generator | None" = None,
        device=None,
        gain_w: float = 1.0,
        gain_b: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        device = get_default_device() if device is None else torch.device(device)
        xkw = dict(rng=rng, device=device, gain_w=gain_w, gain_b=gain_b)
        hkw = dict(rng=rng, device=device, gain_w=gain_w, gain_b=gain_b, bias=False)

        # Input projections (with bias)
        self.Wi = Linear(input_size, hidden_size, **xkw)
        self.Wf = Linear(input_size, hidden_size, **xkw)
        self.Wg = Linear(input_size, hidden_size, **xkw)
        self.Wo = Linear(input_size, hidden_size, **xkw)
        # Hidden (recurrent) projections (no bias)
        self.Ui = Linear(hidden_size, hidden_size, **hkw)
        self.Uf = Linear(hidden_size, hidden_size, **hkw)
        self.Ug = Linear(hidden_size, hidden_size, **hkw)
        self.Uo = Linear(hidden_size, hidden_size, **hkw)

    def forward(
        self, x: GaussianTensor, h: GaussianTensor, c: GaussianTensor
    ) -> tuple[GaussianTensor, GaussianTensor]:
        i = sigmoid(self.Wi(x) + self.Ui(h))
        f = sigmoid(self.Wf(x) + self.Uf(h))
        g = tanh(self.Wg(x) + self.Ug(h))
        o = sigmoid(self.Wo(x) + self.Uo(h))
        c_new = f * c + i * g
        h_new = o * tanh(c_new)
        return h_new, c_new


class LSTM(Module):
    """Bayesian LSTM that unrolls an :class:`LSTMCell` over a sequence.

    ``forward`` takes a python list of per-timestep :class:`GaussianTensor`s,
    each of shape ``(B, input_size)``, and returns the final hidden state
    ``(B, hidden_size)`` — or the list of all hidden states when
    ``return_sequence=True``. The cell (and thus its weights) is shared across
    timesteps.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rng: "torch.Generator | None" = None,
        device=None,
        gain_w: float = 1.0,
        gain_b: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = get_default_device() if device is None else torch.device(device)
        self.cell = LSTMCell(
            input_size, hidden_size, rng=rng, device=self.device, gain_w=gain_w, gain_b=gain_b
        )

    def init_state(self, batch: int) -> tuple[GaussianTensor, GaussianTensor]:
        """Zero, deterministic (var=0) initial hidden and cell states."""
        z = torch.zeros(batch, self.hidden_size, device=self.device)
        return tensor(z.clone(), var=0.0, name="h0"), tensor(z.clone(), var=0.0, name="c0")

    def forward(self, seq, return_sequence: bool = False):
        if len(seq) == 0:
            raise ValueError("LSTM.forward expects a non-empty sequence")
        h, c = self.init_state(seq[0].shape[0])
        outputs = []
        for x_t in seq:
            h, c = self.cell(x_t, h, c)
            if return_sequence:
                outputs.append(h)
        return outputs if return_sequence else h

    def __repr__(self):
        return f"LSTM(input_size={self.input_size}, hidden_size={self.hidden_size})"
