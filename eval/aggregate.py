#!/usr/bin/env python3
"""Aggregate eval results into a suite x delay table.

Usage: ``python eval/aggregate.py <run-dir-or-results.jsonl> [...]``

The four-suite average is reported only when all four LIBERO suites are present; a partial run
prints ``--`` instead, so an incomplete sweep can never be mistaken for a finished one.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def load_records(paths) -> list[dict]:
    """Read every results.jsonl under the given files/directories. Malformed lines are skipped."""
    records = []
    for entry in paths:
        entry = pathlib.Path(entry)
        files = [entry] if entry.is_file() else sorted(entry.rglob("results.jsonl"))
        for path in files:
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def suite_delay_table(records) -> dict:
    """{delay: {suite: mean solve_rate}} plus a parallel count table."""
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))
    for row in records:
        cell = acc[int(row["delay"])][row["benchmark"]]
        cell[0] += float(row["solve_rate"])
        cell[1] += 1
    return {d: {s: (v[0] / v[1]) for s, v in suites.items() if v[1]}
            for d, suites in acc.items()}


def episode_counts(records) -> dict:
    acc = collections.defaultdict(lambda: collections.defaultdict(int))
    for row in records:
        acc[int(row["delay"])][row["benchmark"]] += 1
    return {d: dict(s) for d, s in acc.items()}


def four_suite_average(table) -> dict:
    """Unweighted mean over the four suites, or None when any is missing."""
    out = {}
    for delay, suites in table.items():
        present = [suites[s] for s in SUITES if s in suites]
        out[delay] = (sum(present) / len(SUITES)) if len(present) == len(SUITES) else None
    return out


def format_table(table, averages, counts=None) -> str:
    suites = [s for s in SUITES if any(s in v for v in table.values())]
    extra = sorted({s for v in table.values() for s in v} - set(SUITES))
    columns = suites + extra
    width = max([12] + [len(c) for c in columns]) + 2
    lines = ["delay".ljust(7) + "".join(c.rjust(width) for c in columns)
             + "4-suite avg".rjust(width)]
    lines.append("-" * len(lines[0]))
    for delay in sorted(table):
        row = str(delay).ljust(7)
        for column in columns:
            value = table[delay].get(column)
            row += ("--" if value is None else f"{value:.4f}").rjust(width)
        avg = averages.get(delay)
        row += ("--" if avg is None else f"{avg:.4f}").rjust(width)
        lines.append(row)
    if counts:
        lines.append("")
        lines.append("episodes per cell:")
        for delay in sorted(counts):
            cells = ", ".join(f"{s}={n}" for s, n in sorted(counts[delay].items()))
            lines.append(f"  d={delay}: {cells}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="run directories or results.jsonl files")
    args = p.parse_args()
    records = load_records(args.paths)
    if not records:
        print("no records found", file=sys.stderr)
        raise SystemExit(1)
    methods = sorted({r.get("method", "?") for r in records})
    backbones = sorted({r.get("base_model", "?") for r in records})
    table = suite_delay_table(records)
    print(f"{len(records)} episodes | method(s)={methods} | backbone(s)={backbones}\n")
    print(format_table(table, four_suite_average(table), episode_counts(records)))


if __name__ == "__main__":
    main()
