"""The full-state residual corrector network and its checkpoint format.

Vendored from ``action_state_residual_repro/scripts/lerobot_action_state_residual.py``. The on-disk
schema is unchanged, so checkpoints produced by that bundle load here directly.
"""
from __future__ import annotations

import pathlib
from typing import Any

import torch
import torch.nn as nn

RESIDUAL_TYPE = "full_state_relative_rotation"


class FullStateResidualMLP(nn.Module):
    """Predicts the 8D residual between the controller-target proxy and the true state."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 8))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_corrector(path: str | pathlib.Path, model: nn.Module, **metadata: Any) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "metadata": dict(metadata)}, path)


def load_corrector(path: str | pathlib.Path, map_location: str = "cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    meta = payload["metadata"]
    if meta.get("residual_type") != RESIDUAL_TYPE:
        raise ValueError(
            f"{path}: expected residual_type {RESIDUAL_TYPE!r}, got {meta.get('residual_type')!r}")
    model = FullStateResidualMLP(
        input_dim=int(meta["input_dim"]),
        hidden_dim=int(meta.get("hidden_dim", 128)),
        num_layers=int(meta.get("num_layers", 2)),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, meta
