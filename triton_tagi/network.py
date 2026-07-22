"""
Network builder — a Sequential container for TAGI layers.

Supports both MLP and CNN architectures:

    # MLP
    net = Sequential([
        Linear(784, 256), ReLU(),
        Linear(256, 10),  Remax(),
    ])

    # CNN
    net = Sequential([
        Conv2D(1, 32, 5, padding=2), ReLU(), AvgPool2D(2),
        Conv2D(32, 64, 5, padding=2), ReLU(), AvgPool2D(2),
        Flatten(),
        Linear(3136, 256), ReLU(),
        Linear(256, 10),   Remax(),
    ])

The step() method follows cuTAGI's architecture:
    1. Forward pass — propagate moments
    2. Compute output innovation
    3. Backward pass — compute and store deltas on each layer (NO update)
    4. Update — apply capped deltas to all learnable layers
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import Layer, LearnableLayer
from .layers.multihead_attention import MultiheadAttentionV2
from .layers.resblock import ResBlock
from .update.observation import compute_innovation, compute_innovation_with_indices
from .update.parameters import get_cap_factor


class Sequential:
    """
    Sequential container for TAGI Bayesian neural networks.

    Parameters
    ----------
    layers : list of layer objects
    device : str or torch.device  (default "cpu")
    """

    def __init__(self, layers: list, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.layers = layers

        # Move learnable layers to the target device
        for layer in self.layers:
            if isinstance(layer, ResBlock):
                # These blocks manage their own sub-layers
                layer.device = self.device
                for sub in layer._learnable:
                    self._move_layer_to_device(sub)
            elif isinstance(layer, LearnableLayer):
                self._move_layer_to_device(layer)
            # Move BatchNorm running stats
            if hasattr(layer, "running_mean"):
                layer.running_mean = layer.running_mean.to(self.device)
                layer.running_var = layer.running_var.to(self.device)

    def _move_layer_to_device(self, layer):
        """Move a single layer's parameters to self.device."""
        if isinstance(layer, MultiheadAttentionV2):
            layer.device = self.device
            for sub in (layer.q_proj, layer.k_proj, layer.v_proj):
                self._move_layer_to_device(sub)
            return
        if not hasattr(layer, "mw") or layer.mw is None:
            return
        layer.device = self.device
        layer.mw = layer.mw.to(self.device)
        if getattr(layer, "Sw", None) is not None:
            layer.Sw = layer.Sw.to(self.device)
        if getattr(layer, "mb", None) is not None:
            layer.mb = layer.mb.to(self.device)
            if getattr(layer, "Sb", None) is not None:
                layer.Sb = layer.Sb.to(self.device)
        if getattr(layer, "running_mean", None) is not None:
            layer.running_mean = layer.running_mean.to(self.device)
            layer.running_var = layer.running_var.to(self.device)

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass through the entire network.

        Parameters
        ----------
        x : Tensor  input data (flat or spatial)

        Returns
        -------
        mu  : Tensor  predicted output means
        var : Tensor  predicted output variances
        """
        ma = x
        Sa = torch.zeros_like(x)

        for layer in self.layers:
            if isinstance(layer, Layer):
                ma, Sa = layer.forward(ma, Sa)
            else:
                raise TypeError(f"Unknown layer type: {type(layer)}")

        return ma, Sa

    # ------------------------------------------------------------------
    #  Single training step (cuTAGI-style: backward + capped update)
    # ------------------------------------------------------------------
    def step(self, x_batch: Tensor, y_batch: Tensor, sigma_v: float) -> tuple[Tensor, Tensor]:
        """
        Perform one forward + backward + capped-update TAGI step.

        Parameters
        ----------
        x_batch : Tensor  input mini-batch
        y_batch : Tensor  target mini-batch
        sigma_v : float   observation noise std

        Returns
        -------
        y_pred_mu  : Tensor  predicted means (before update)
        y_pred_var : Tensor  predicted variances (before update)
        """
        batch_size = x_batch.shape[0]

        # ── 1. Forward ──
        y_pred_mu, y_pred_var = self.forward(x_batch)

        # ── 2. Output innovation ──
        delta_mu, delta_var = compute_innovation(y_batch, y_pred_mu, y_pred_var, sigma_v)

        # ── 3. Backward (compute + store deltas, NO param update) ──
        for layer in reversed(self.layers):
            delta_mu, delta_var = layer.backward(delta_mu, delta_var)

        # ── 4. Capped parameter update (cuTAGI-style) ──
        cap_factor = get_cap_factor(batch_size)
        for layer in self.layers:
            if isinstance(layer, LearnableLayer):
                layer.update(cap_factor)

        return y_pred_mu, y_pred_var

    # ------------------------------------------------------------------
    #  Hierarchical softmax training step
    # ------------------------------------------------------------------
    def step_hrc(
        self,
        x_batch: Tensor,
        labels: Tensor,
        hrc: "HierarchicalSoftmax",
        sigma_v: float,
    ) -> tuple[Tensor, Tensor]:
        """One forward + backward + capped-update step using hierarchical softmax.

        Uses the sparse output innovation from :func:`compute_innovation_with_indices`
        so that only the ``n_obs`` tree nodes on each class's binary path receive
        an update signal, matching cuTAGI's ``update_using_indices``.

        The output layer must have ``hrc.len`` output neurons::

            net = Sequential([..., Linear(hidden, hrc.len)])

        Args:
            x_batch: Input mini-batch, shape (B, in_features).
            labels:  Integer class labels, shape (B,).
            hrc:     HierarchicalSoftmax from :func:`triton_tagi.hrc_softmax.class_to_obs`.
            sigma_v: Observation noise standard deviation.

        Returns:
            y_pred_mu:  Predicted output means before update, shape (B, hrc.len).
            y_pred_var: Predicted output variances before update, shape (B, hrc.len).
        """
        from .hrc_softmax import labels_to_hrc

        batch_size = x_batch.shape[0]

        # 1. Forward pass. Sequence models output (B, S, hrc.len); flatten to
        #    (B*S, hrc.len) for innovation, then reshape the delta back.
        y_pred_mu, y_pred_var = self.forward(x_batch)
        pred_shape = y_pred_mu.shape
        if y_pred_mu.dim() == 3:
            ma_flat = y_pred_mu.reshape(-1, pred_shape[-1])
            Sa_flat = y_pred_var.reshape(-1, pred_shape[-1])
        else:
            ma_flat = y_pred_mu
            Sa_flat = y_pred_var

        # 2. Encode labels → (obs ±1, 1-indexed node positions)
        y_obs, y_idx = labels_to_hrc(labels, hrc)

        # var_obs: scalar sigma_v^2 broadcast to (N, n_obs)
        var_obs = torch.full_like(y_obs, sigma_v**2)

        # 3. Sparse output innovation
        delta_mu, delta_var = compute_innovation_with_indices(
            ma_flat, Sa_flat, y_obs, var_obs, y_idx
        )

        if y_pred_mu.dim() == 3:
            delta_mu = delta_mu.reshape(pred_shape)
            delta_var = delta_var.reshape(pred_shape)

        # 4. Backward pass (identical to dense step)
        for layer in reversed(self.layers):
            delta_mu, delta_var = layer.backward(delta_mu, delta_var)

        # 5. Capped parameter update
        cap_factor = get_cap_factor(batch_size)
        for layer in self.layers:
            if isinstance(layer, LearnableLayer):
                layer.update(cap_factor)

        return y_pred_mu, y_pred_var

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Set all layers to training mode (affects BatchNorm, etc.)."""
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.train()

    def eval(self) -> None:
        """Set all layers to evaluation mode (affects BatchNorm, etc.)."""
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.eval()

    def __repr__(self):
        lines = ["Sequential("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}): {layer}")
        lines.append(")")
        return "\n".join(lines)

    def num_parameters(self) -> int:
        """Return total number of learnable scalars (means + variances)."""
        return sum(
            layer.num_parameters for layer in self.layers if isinstance(layer, LearnableLayer)
        )

    def get_attention_scores(self) -> dict[int, tuple[Tensor, Tensor]]:
        """Collect attention score moments (μ, var) from every attention layer.

        Returns an ordered dict keyed by the layer's position in ``self.layers``.
        Each value is ``(mu_score, var_score)`` of shape ``(B, H, S, S)`` from
        the most recent forward pass. Raises if no attention layer has run.
        """
        out: dict[int, tuple[Tensor, Tensor]] = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, MultiheadAttentionV2):
                out[i] = layer.get_attention_scores()
        if not out:
            raise RuntimeError("No attention layers in this Sequential.")
        return out
