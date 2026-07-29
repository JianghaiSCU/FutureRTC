"""Per-evaluation record schema, aggregation, and output writers.

One record is a single evaluated Kinetix point: a ``(level, delay, execute_horizon)``
whose ``solve_rate`` is the mean over ``n_trials`` and whose ``execution_time`` is the mean
episode length in env steps. ``summarize`` collapses the horizon sweep to per-delay means;
``per_task_summary`` keeps the per-level breakdown.
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
    "benchmark",
    "base_model",
    "method",
    "delay",
    "execute_horizon",
    "task_or_level_id",
    "episode_idx",
    "solve_rate",
    "n_trials",
    "execution_time",
    "seed",
]


def to_row(record: EpisodeRecord) -> dict:
    """Convert a record to a dict keyed by FIELDNAMES (stable column order)."""
    data = dataclasses.asdict(record)
    return {name: data[name] for name in FIELDNAMES}


def build_point_record(
    *, method, delay, execute_horizon, level_name, solve_rate, n_trials, execution_time, seed,
    base_model="bc31",
) -> EpisodeRecord:
    """Build one EpisodeRecord for a Kinetix (level, method, delay, horizon) point."""
    return EpisodeRecord(
        benchmark="kinetix",
        base_model=base_model,
        method=method,
        delay=delay,
        task_or_level_id=level_name,
        episode_idx=0,
        solve_rate=float(solve_rate),
        n_trials=int(n_trials),
        execution_time=float(execution_time),
        seed=seed,
        execute_horizon=execute_horizon,
    )


def summarize(records: Iterable[EpisodeRecord]) -> list[dict]:
    """Aggregate by (benchmark, base_model, method, delay), averaging over records."""
    grouped: dict[tuple, list[EpisodeRecord]] = {}
    for item in records:
        key = (item.benchmark, item.base_model, item.method, item.delay)
        grouped.setdefault(key, []).append(item)

    rows: list[dict] = []
    for (benchmark, base_model, method, delay), items in grouped.items():
        n = len(items)
        rows.append(
            {
                "benchmark": benchmark,
                "base_model": base_model,
                "method": method,
                "delay": delay,
                "average_solve_rate": sum(x.solve_rate for x in items) / n,
                "average_execution_time": sum(x.execution_time for x in items) / n,
                "num_level_horizon_points": n,
            }
        )
    return rows


def per_task_summary(records: Iterable[EpisodeRecord]) -> list[dict]:
    """Per-level aggregation: mean over the horizon sweep, keeping each level separate."""
    grouped: dict[tuple, list[EpisodeRecord]] = {}
    for item in records:
        key = (item.benchmark, item.base_model, item.method, item.delay, item.task_or_level_id)
        grouped.setdefault(key, []).append(item)

    rows: list[dict] = []
    for (benchmark, base_model, method, delay, task), items in grouped.items():
        n = len(items)
        rows.append(
            {
                "benchmark": benchmark,
                "base_model": base_model,
                "method": method,
                "delay": delay,
                "task_or_level_id": task,
                "solve_rate": sum(x.solve_rate for x in items) / n,
                "execution_time": sum(x.execution_time for x in items) / n,
                "num_horizon_points": n,
            }
        )
    return rows


def write_outputs(output_dir, records) -> None:
    """Write results.jsonl (one row per point) and summary.csv (per-delay means)."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "results.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(to_row(rec)) + "\n")

    summary = summarize(records)
    if summary:
        fieldnames = ["benchmark", "base_model", "method", "delay", "average_solve_rate",
                      "average_execution_time", "num_level_horizon_points"]
        with (output_dir / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary)
