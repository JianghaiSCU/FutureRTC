"""Delay / execute-horizon sweep enumeration for the Kinetix delay-robustness eval.

Pure functions, no policy / GPU / simulator. Mirrors RTC's
``src/eval_oracle_delay_compare.py``: for each inference delay ``d``, the execute horizon
ranges over ``[max(1, d), action_chunk_size - d]``.
"""
from __future__ import annotations

from typing import Sequence


def build_delay_horizon_pairs(
    action_chunk_size: int, delays: Sequence[int] = (1, 2, 3, 4)
) -> list[tuple[int, int]]:
    """Enumerate (delay, execute_horizon) pairs for the sweep."""
    pairs: list[tuple[int, int]] = []
    for delay in delays:
        if delay < 0:
            raise ValueError(f"delay must be non-negative, got {delay}")
        for execute_horizon in range(max(1, delay), action_chunk_size - delay + 1):
            pairs.append((delay, execute_horizon))
    return pairs


def limit_horizon_pairs_per_delay(
    pairs: Sequence[tuple[int, int]], max_horizons_per_delay: int | None
) -> list[tuple[int, int]]:
    """Keep at most ``max_horizons_per_delay`` pairs for each delay (preserves order)."""
    if max_horizons_per_delay is None:
        return list(pairs)
    if max_horizons_per_delay <= 0:
        raise ValueError(
            f"max_horizons_per_delay must be positive, got {max_horizons_per_delay}"
        )
    counts: dict[int, int] = {}
    limited: list[tuple[int, int]] = []
    for delay, execute_horizon in pairs:
        if counts.get(delay, 0) < max_horizons_per_delay:
            limited.append((delay, execute_horizon))
            counts[delay] = counts.get(delay, 0) + 1
    return limited
