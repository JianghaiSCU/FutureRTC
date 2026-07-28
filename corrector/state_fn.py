"""Corrector-produced proprioceptive state.

Turns the measured state at the handoff-minus-delay frame plus the ``d`` committed actions into an
estimate of the state at the handoff. Two entry points share one implementation:

* ``Corrector.correct_batch`` -- batched, used by the eval driver and the policy loss.
* ``CorrectorStateFn``        -- per-sample over a resident state array, used as the predictor
                                 dataset's ``state_fn``. CPU-only, so it is safe to fork into
                                 DataLoader workers.

ACTION SPACE -- read before touching this file. The corrector forward-integrates PHYSICAL OSC
deltas, so ``Corrector.correct_batch`` accepts ENV-space actions ONLY. Callers holding the latent
bank's quantile-normalized actions must convert with ``common.action_space.qnorm_to_env`` FIRST and
only then right-align them into the ``[n, max_delay, 7]`` buffer.

The order matters and is not interchangeable: an ENV-space zero is a true no-op (no translation, no
rotation, ``sign(0) == 0`` leaves the gripper alone), whereas a BANK-space zero un-qnorms to the
MIDPOINT of the action quantile range -- a real motion. Converting the padded buffer instead of the
real slice therefore injects a spurious action on every padded step. ``CorrectorStateFn`` does it in
the right order; that is why the conversion lives there and not inside ``correct_batch``.
"""
from __future__ import annotations

import numpy as np
import torch

from common.action_space import load_action_stats, qnorm_to_env
from corrector.model import load_corrector
from corrector.physics import (
    apply_full_state_residual, build_full_state_residual_features,
    denormalize_full_state_residual, make_controller_target_state_canonical_prev,
)


class Corrector:
    """Frozen full-state residual corrector. Actions are ENV space (physical OSC deltas)."""

    def __init__(self, ckpt_path):
        self.model, self.metadata = load_corrector(ckpt_path)
        self.residual_scale = np.asarray(self.metadata["residual_scale"], dtype=np.float32)
        self.max_delay = int(self.metadata.get("max_delay", 20))

    @torch.no_grad()
    def correct_batch(self, base_state8, committed_actions_env, delay: int) -> np.ndarray:
        """base_state8 [n,8]; committed_actions_env [n,K,7] ENV space, the last ``delay`` entries
        being the committed actions and any earlier entries zero; delay in [0, max_delay]."""
        delay = int(delay)
        if not 0 <= delay <= self.max_delay:
            raise ValueError(f"delay must be in [0, {self.max_delay}], got {delay}")
        base = np.asarray(base_state8, dtype=np.float32)
        acts = np.asarray(committed_actions_env, dtype=np.float32)
        proxy = make_controller_target_state_canonical_prev(base, acts)
        feats = build_full_state_residual_features(base, acts, proxy, delay,
                                                   max_delay=self.max_delay)
        residual = self.model(torch.from_numpy(feats)).numpy()
        residual = denormalize_full_state_residual(residual, self.residual_scale)
        return apply_full_state_residual(proxy, residual).astype(np.float32)


def right_align_env_actions(committed_actions, delay: int, d_max: int, *,
                            action_space: str = "qnorm", stats=None) -> np.ndarray:
    """Build the ``[1, d_max, 7]`` ENV-space buffer ``Corrector.correct_batch`` expects.

    Converts the ``delay`` real actions out of BANK space FIRST, then pads with ENV zeros, so the
    padding stays a true no-op (see this module's docstring).
    """
    if action_space not in ("qnorm", "env"):
        raise ValueError(f"action_space must be 'qnorm' or 'env', got {action_space!r}")
    acts_ra = np.zeros((1, d_max, 7), dtype=np.float32)
    delay = int(delay)
    if delay > 0:
        raw = (committed_actions.detach().cpu().numpy()
               if isinstance(committed_actions, torch.Tensor) else committed_actions)
        acts = np.asarray(raw, dtype=np.float32)[:delay]
        if action_space == "qnorm":
            acts = np.asarray(qnorm_to_env(acts, stats), dtype=np.float32)
        acts_ra[0, -delay:] = acts
    return acts_ra


class CorrectorStateFn:
    """``(g0, d, committed_actions[d,7]) -> corrected state [8]`` over a resident state array."""

    def __init__(self, ckpt_path, state_array, *, d_max: int = 20, action_stats_path=None,
                 action_space: str = "qnorm"):
        if action_space not in ("qnorm", "env"):
            raise ValueError(f"action_space must be 'qnorm' or 'env', got {action_space!r}")
        self.corrector = Corrector(ckpt_path)
        self.state_np = np.asarray(state_array, dtype=np.float32)
        self.d_max = int(d_max)
        self.action_space = action_space
        self.stats = load_action_stats(action_stats_path)

    def __call__(self, g0: int, d: int, committed_actions) -> torch.Tensor:
        base = self.state_np[int(g0), :8][None, :]
        acts_ra = right_align_env_actions(committed_actions, d, self.d_max,
                                          action_space=self.action_space, stats=self.stats)
        corrected = self.corrector.correct_batch(base, acts_ra, int(d))
        return torch.from_numpy(corrected[0]).float()
