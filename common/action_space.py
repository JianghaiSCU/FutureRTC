"""LIBERO action-space conversions.

Three spaces, never to be mixed (a past mix-up cost 10-30x errors on the rotation dims):

* BANK   -- what the latent bank stores: ``qnorm(env) = 2*(a-q01)/(q99-q01) - 1``.
* ENV    -- physical OSC deltas; what the corrector forward-integrates and what the
            eval postprocessor emits.
* POLICY -- ``(a_env - mean)/std``; both backbones normalize ACTION with MEAN_STD, so this
            is the space the flow head's actions and velocities live in.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import torch

DEFAULT_STATS_PATH = pathlib.Path(__file__).resolve().parents[1] / "assets" / "libero_action_stats.json"


@dataclasses.dataclass(frozen=True)
class ActionStats:
    q01: np.ndarray
    q99: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def load_action_stats(path: str | pathlib.Path | None = None) -> ActionStats:
    payload = json.loads(pathlib.Path(path or DEFAULT_STATS_PATH).read_text())
    return ActionStats(**{k: np.asarray(payload[k], dtype=np.float32)
                          for k in ("q01", "q99", "mean", "std")})


def _like(a, ref):
    """Return ``a`` (numpy) matched to ``ref``'s container type/device/dtype."""
    if isinstance(ref, torch.Tensor):
        dtype = ref.dtype if ref.is_floating_point() else torch.float32
        return torch.as_tensor(a, dtype=dtype, device=ref.device)
    return np.asarray(a, dtype=np.float32)


def _np(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().float().numpy()
    return np.asarray(a, dtype=np.float32)


def qnorm_to_env(a, stats: ActionStats):
    """BANK -> ENV. Inverse of the collector's qnorm."""
    x = _np(a)
    d = x.shape[-1]
    out = stats.q01[:d] + (x + 1.0) * (stats.q99[:d] - stats.q01[:d]) / 2.0
    return _like(out, a)


def env_to_qnorm(a, stats: ActionStats):
    """ENV -> BANK."""
    x = _np(a)
    d = x.shape[-1]
    out = 2.0 * (x - stats.q01[:d]) / (stats.q99[:d] - stats.q01[:d]) - 1.0
    return _like(out, a)


def qnorm_to_policy(a, stats: ActionStats):
    """BANK -> POLICY (MEAN_STD), i.e. the space the flow head operates in."""
    x = _np(qnorm_to_env(a, stats))
    d = x.shape[-1]
    out = (x - stats.mean[:d]) / stats.std[:d]
    return _like(out, a)
