"""(state, committed actions, delay, ground-truth state) triples for corrector training.

The corrector needs only state and action, so this pulls those columns out of the already-loaded
LeRobot ``hf_dataset`` and never decodes an image. Delay is resampled on every access, giving full
(frame x delay) coverage from a fixed index.

ACTION SPACE: the dataset's ``action`` column is ENV space (physical OSC deltas) -- exactly what
the corrector integrates -- so no conversion happens here. The quantile normalization only enters
on the predictor side, where actions come from the latent bank.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from common.run_index import build_index, contiguous_run_lengths
from corrector.physics import (
    build_full_state_residual_features, build_full_state_residual_targets,
    make_controller_target_state_canonical_prev,
)


class ResidualDataset(torch.utils.data.Dataset):
    """Yields the corrector's (features, normalized-target, proxy, gt, delay) tuple.

    ``d ~ Uniform{0..min(d_max, room)}`` where ``room`` is the number of in-run future steps, so a
    pair never crosses an episode or a gap in the locally-cached frames.
    """

    def __init__(self, states, actions, run_lengths, *, d_max: int = 20, seed: int = 0):
        self.states = np.asarray(states, dtype=np.float32)[:, :8]
        self.actions = np.asarray(actions, dtype=np.float32)
        self.run_lengths = list(run_lengths)
        self.d_max = int(d_max)
        self._rng = random.Random(seed)
        self.index = build_index(self.run_lengths, self.d_max)

    @classmethod
    def from_lerobot(cls, lerobot_dataset, *, d_max: int = 20, seed: int = 0):
        sub = lerobot_dataset.hf_dataset.select_columns(
            ["observation.state", "action", "episode_index", "frame_index"]).with_format("numpy")
        return cls(
            np.asarray(sub["observation.state"], dtype=np.float32),
            np.asarray(sub["action"], dtype=np.float32),
            contiguous_run_lengths(sub["episode_index"], sub["frame_index"]),
            d_max=d_max, seed=seed,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        run_id, local_t, run_start = self.index[i]
        room = self.run_lengths[run_id] - 1 - local_t
        d = self._rng.randint(0, min(self.d_max, room))
        g0 = run_start + local_t
        base = self.states[g0]
        gt = self.states[g0 + d]
        acts_ra = np.zeros((self.d_max, 7), dtype=np.float32)  # right-aligned committed actions
        if d > 0:
            acts_ra[-d:] = self.actions[g0:g0 + d]
        proxy = make_controller_target_state_canonical_prev(base[None], acts_ra[None])[0]
        target = build_full_state_residual_targets(proxy[None], gt[None])[0]
        features = build_full_state_residual_features(
            base[None], acts_ra[None], proxy[None], d, max_delay=self.d_max)[0]
        return {
            "features": torch.from_numpy(np.asarray(features, dtype=np.float32)),
            "target": torch.from_numpy(np.asarray(target, dtype=np.float32)),
            "proxy": torch.from_numpy(np.asarray(proxy, dtype=np.float32)),
            "gt": torch.from_numpy(np.asarray(gt, dtype=np.float32)),
            "delay": torch.tensor(d, dtype=torch.long),
        }
