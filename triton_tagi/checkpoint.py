"""
Run directory management, checkpointing, and metrics logging.

Every training run produces artifacts (checkpoints, metrics, figures) under a
single run directory rooted at ``runs/``. The directory name encodes the three
dimensions that distinguish runs::

    runs/{dataset}_{arch}_{optimizer}_{YYYYMMDD-HHMMSS}/

``RunDir`` is the write-side API: examples create one at the start of training
and use it for all I/O. ``load_model`` is the read-side convenience: pass it a
checkpoint path and a constructor that builds the network topology, and it
returns a ready-to-use ``Sequential`` with the saved parameters restored.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch

from .base import LearnableLayer
from .network import Sequential


class RunDir:
    """Manages the directory structure for a single training run.

    Args:
        dataset:   Dataset name, e.g. ``"mnist"``, ``"cifar10"``.
        arch:      Architecture name, e.g. ``"mlp"``, ``"resnet18"``.
        optimizer: Optimizer name, e.g. ``"tagi"``.
        base:      Root directory for all runs (default ``"runs"``).
    """

    def __init__(
        self,
        dataset: str,
        arch: str,
        optimizer: str,
        base: str = "runs",
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{dataset}_{arch}_{optimizer}_{timestamp}"
        self.path = Path(base) / name
        self.checkpoints = self.path / "checkpoints"
        self.figures = self.path / "figures"
        self.config_json = self.path / "config.json"
        self.metrics_csv = self.path / "metrics.csv"
        self.path.mkdir(parents=True, exist_ok=True)
        self.checkpoints.mkdir()
        self.figures.mkdir()

    def save_config(self, config: dict[str, Any]) -> None:
        """Write config.json. Call before the first training step."""
        with open(self.config_json, "w") as f:
            json.dump(config, f, indent=2)

    def save_checkpoint(
        self,
        net: Sequential,
        epoch: int,
        config: dict[str, Any],
    ) -> Path:
        """Save a checkpoint to ``checkpoints/epoch_{epoch:04d}.pt``."""
        ck = {
            "epoch": epoch,
            "config": config,
            "net_state": _extract_net_state(net),
        }
        path = self.checkpoints / f"epoch_{epoch:04d}.pt"
        torch.save(ck, path)
        return path

    def load_checkpoint(
        self,
        net: Sequential,
        path: str | Path | None = None,
    ) -> int:
        """Load a checkpoint into ``net`` and return the saved epoch number.

        If ``path`` is ``None``, loads the latest ``epoch_*.pt`` in
        ``self.checkpoints/``.
        """
        if path is None:
            candidates = sorted(self.checkpoints.glob("epoch_*.pt"))
            if not candidates:
                raise FileNotFoundError(f"No checkpoints found in {self.checkpoints}")
            path = candidates[-1]

        ck = torch.load(path, map_location=net.device, weights_only=False)
        _restore_net_state(net, ck["net_state"])
        return ck["epoch"]

    def append_metrics(self, epoch: int, **kwargs: float) -> None:
        """Append one row to ``metrics.csv``.

        Creates the file with a header derived from the first call's keys.
        Subsequent calls must use the same keyword arguments.
        """
        row = {"epoch": epoch, **kwargs}
        write_header = not self.metrics_csv.exists()
        with open(self.metrics_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow({k: f"{v:.6g}" if isinstance(v, float) else v for k, v in row.items()})

    def __repr__(self) -> str:
        return f"RunDir({self.path})"


def load_model(
    path: str | Path,
    build: Callable[[dict[str, Any]], Sequential],
    device: str | torch.device = "cpu",
) -> tuple[Sequential, dict[str, Any], int]:
    """Load a trained model from a checkpoint file.

    Args:
        path:   Path to an ``epoch_*.pt`` file saved by :meth:`RunDir.save_checkpoint`.
        build:  Callable that takes the saved ``config`` dict and returns an
                un-initialised ``Sequential`` with the same topology used at
                training time. The network's parameter tensors are overwritten
                with the saved values.
        device: Device to place the restored network on.

    Returns:
        (net, config, epoch) — the restored network, its training config, and
        the epoch number saved in the checkpoint.

    Example::

        from triton_tagi import load_model, Sequential, Linear, ReLU

        def build(cfg):
            return Sequential(
                Linear(cfg["in"], 128),
                ReLU(),
                Linear(128, cfg["out"]),
                device="cpu",
            )

        net, cfg, epoch = load_model("runs/.../checkpoints/epoch_0100.pt", build)
        mu, var = net.step(x, eval_mode=True)
    """
    device = torch.device(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    config = ck["config"]
    net = build(config)
    if hasattr(net, "to"):
        net = net.to(device)
    _restore_net_state(net, ck["net_state"])
    return net, config, ck["epoch"]


# ---------------------------------------------------------------------------
#  Internal helpers — network state
# ---------------------------------------------------------------------------

_PARAM_ATTRS = ("mw", "Sw", "mb", "Sb")
_STAT_ATTRS = ("running_mean", "running_var")


def _layer_state(layer: object) -> dict[str, torch.Tensor]:
    """Extract serialisable parameter tensors from a single leaf layer."""
    d = {}
    for attr in _PARAM_ATTRS + _STAT_ATTRS:
        if hasattr(layer, attr):
            d[attr] = getattr(layer, attr).detach().cpu().clone()
    return d


def _extract_net_state(net: Sequential) -> dict[int, Any]:
    """Extract state from all learnable layers, keyed by position in ``net.layers``.

    Layers with a ``_learnable`` attribute (ResBlock-style) are stored as a list
    of per-sublayer dicts. Simple ``LearnableLayer`` layers are stored as a plain
    dict.
    """
    state: dict[int, Any] = {}
    for i, layer in enumerate(net.layers):
        if hasattr(layer, "_learnable"):
            sub_states = [_layer_state(sub) for sub in layer._learnable]
            if any(sub_states):
                state[i] = sub_states
        elif isinstance(layer, LearnableLayer):
            s = _layer_state(layer)
            if s:
                state[i] = s
    return state


def _restore_layer(layer: object, d: dict[str, torch.Tensor], device: torch.device) -> None:
    for attr in _PARAM_ATTRS + _STAT_ATTRS:
        if attr in d and hasattr(layer, attr):
            getattr(layer, attr).data.copy_(d[attr].to(device))


def _restore_net_state(net: Sequential, state: dict[int, Any]) -> None:
    device = net.device
    for i, layer in enumerate(net.layers):
        if i not in state:
            continue
        s = state[i]
        if isinstance(s, list):
            for sub, sub_state in zip(layer._learnable, s):
                _restore_layer(sub, sub_state, device)
        else:
            _restore_layer(layer, s, device)
