"""Per-episode records, aggregation, and result output.

A record is one evaluated LIBERO episode, so ``solve_rate`` is 0.0 or 1.0 and ``n_trials`` is 1.
``execution_time`` is the episode length in env steps.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import pathlib
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
    benchmark: str
    base_model: str
    method: str
    delay: int
    task_or_level_id: str
    episode_idx: int
    solve_rate: float
    n_trials: int
    execution_time: float
    seed: int
    execute_horizon: int | None = None


# Canonical column order for results.jsonl.
FIELDNAMES = [
    "benchmark", "base_model", "method", "delay", "execute_horizon", "task_or_level_id",
    "episode_idx", "solve_rate", "n_trials", "execution_time", "seed",
]

SUMMARY_FIELDNAMES = [
    "benchmark", "base_model", "method", "delay", "average_solve_rate",
    "average_execution_time", "num_episodes",
]


def to_row(record: EpisodeRecord) -> dict:
    """Convert a record to a dict keyed by FIELDNAMES (stable column order)."""
    data = dataclasses.asdict(record)
    return {name: data[name] for name in FIELDNAMES}


def summarize(records: Iterable[EpisodeRecord]) -> list[dict]:
    """Aggregate by (benchmark, base_model, method, delay), averaging over episodes."""
    grouped: dict[tuple, list[EpisodeRecord]] = {}
    for item in records:
        key = (item.benchmark, item.base_model, item.method, item.delay)
        grouped.setdefault(key, []).append(item)

    rows: list[dict] = []
    for (benchmark, base_model, method, delay), items in grouped.items():
        n = len(items)
        rows.append({
            "benchmark": benchmark,
            "base_model": base_model,
            "method": method,
            "delay": delay,
            "average_solve_rate": sum(x.solve_rate for x in items) / n,
            "average_execution_time": sum(x.execution_time for x in items) / n,
            "num_episodes": n,
        })
    return rows


def write_outputs(output_dir, records) -> None:
    """Write results.jsonl (one record per line) and summary.csv."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "results.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(to_row(record)) + "\n")

    summary = summarize(records)
    if summary:
        with (output_dir / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
            writer.writeheader()
            writer.writerows(summary)
