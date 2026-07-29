"""Record schema + aggregation + output-writer tests (pure, no deps)."""
import json
import pathlib
import sys
import tempfile
import unittest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from motion_prior_handoff.results import (  # noqa: E402
    FIELDNAMES,
    build_point_record,
    per_task_summary,
    summarize,
    to_row,
    write_outputs,
)


def _records():
    # two levels x two horizons at delay 1, plus one point at delay 2
    recs = []
    for level, sr in [("a", 0.8), ("b", 0.6)]:
        for h, sr_h in [(1, sr), (2, sr + 0.1)]:
            recs.append(build_point_record(
                method="futurertc", delay=1, execute_horizon=h, level_name=level,
                solve_rate=sr_h, n_trials=2048, execution_time=90.0, seed=0))
    recs.append(build_point_record(
        method="futurertc", delay=2, execute_horizon=2, level_name="a",
        solve_rate=0.7, n_trials=2048, execution_time=95.0, seed=0))
    return recs


class ResultsTest(unittest.TestCase):
    def test_to_row_has_canonical_columns(self):
        row = to_row(_records()[0])
        self.assertEqual(list(row.keys()), FIELDNAMES)
        self.assertEqual(row["method"], "futurertc")
        self.assertEqual(row["base_model"], "bc31")

    def test_summarize_means_over_horizons_and_levels(self):
        summary = {(r["delay"]): r for r in summarize(_records())}
        # delay 1: solve rates 0.8,0.9,0.6,0.7 -> mean 0.75 over 4 points
        self.assertAlmostEqual(summary[1]["average_solve_rate"], 0.75)
        self.assertEqual(summary[1]["num_level_horizon_points"], 4)
        self.assertAlmostEqual(summary[2]["average_solve_rate"], 0.7)

    def test_per_task_keeps_levels_separate(self):
        rows = {(r["task_or_level_id"], r["delay"]): r for r in per_task_summary(_records())}
        # level a, delay 1: 0.8 and 0.9 -> 0.85
        self.assertAlmostEqual(rows[("a", 1)]["solve_rate"], 0.85)
        self.assertAlmostEqual(rows[("b", 1)]["solve_rate"], 0.65)

    def test_write_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            write_outputs(d, _records())
            lines = (pathlib.Path(d) / "results.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(lines), len(_records()))
            self.assertEqual(list(json.loads(lines[0]).keys()), FIELDNAMES)
            self.assertTrue((pathlib.Path(d) / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
