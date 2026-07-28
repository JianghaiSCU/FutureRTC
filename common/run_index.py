"""Contiguous-run structure over a LeRobot frame table.

Both the corrector's (state, actions, delay, gt) triples and the per-frame latent bank's (t, t+d)
pairs must stay inside a single episode AND inside a contiguous stretch of the frames actually
present, so a partial local dataset cache simply yields fewer runs instead of silently pairing
across a gap.
"""
from __future__ import annotations

import numpy as np


def contiguous_run_lengths(episode_index, frame_index) -> list[int]:
    """Lengths of maximal runs of rows that are same-episode and frame-index contiguous (+1)."""
    ei = np.asarray(episode_index)
    fi = np.asarray(frame_index)
    n = len(ei)
    if n == 0:
        return []
    runs = []
    run_len = 1
    for i in range(1, n):
        if ei[i] == ei[i - 1] and fi[i] == fi[i - 1] + 1:
            run_len += 1
        else:
            runs.append(run_len)
            run_len = 1
    runs.append(run_len)
    return runs


def build_index(run_lengths, d_max: int) -> list[tuple[int, int, int]]:
    """Every frame with at least one in-run future step, as (run_id, local_t, run_start_global).

    ``d_max`` is part of the signature for call-site symmetry; the per-sample delay is clamped to
    the available room at access time, so no filtering happens here.
    """
    index = []
    start = 0
    for run_id, length in enumerate(run_lengths):
        for local_t in range(max(0, length - 1)):
            index.append((run_id, local_t, start))
        start += length
    return index
