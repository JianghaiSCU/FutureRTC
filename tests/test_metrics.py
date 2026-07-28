import csv
import json
import pathlib
import tempfile
import unittest

from eval.metrics import EpisodeRecord, summarize, to_row, write_outputs


def _rec(delay, solve, task="0", ep=0):
    return EpisodeRecord(benchmark="libero_spatial", base_model="pi05",
                         method="predictor_corrector", delay=delay, execute_horizon=None,
                         task_or_level_id=task, episode_idx=ep, solve_rate=solve, n_trials=1,
                         execution_time=12.0, seed=7)


class MetricsTest(unittest.TestCase):
    def test_row_keys(self):
        row = to_row(_rec(5, 1.0))
        for key in ("benchmark", "base_model", "method", "delay", "task_or_level_id",
                    "episode_idx", "solve_rate", "n_trials", "execution_time", "seed"):
            self.assertIn(key, row)

    def test_summarize_groups_by_benchmark_model_method_delay(self):
        rows = summarize([_rec(5, 1.0), _rec(5, 0.0), _rec(10, 1.0)])
        self.assertEqual(len(rows), 2)
        by_delay = {r["delay"]: r for r in rows}
        self.assertAlmostEqual(by_delay[5]["average_solve_rate"], 0.5)
        self.assertAlmostEqual(by_delay[10]["average_solve_rate"], 1.0)

    def test_summarize_counts_episodes(self):
        rows = summarize([_rec(5, 1.0, ep=0), _rec(5, 0.0, ep=1)])
        self.assertEqual(rows[0]["num_episodes"], 2)

    def test_write_outputs_emits_both_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            write_outputs(out, [_rec(5, 1.0), _rec(5, 0.0)])
            lines = (out / "results.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["method"], "predictor_corrector")
            with (out / "summary.csv").open() as f:
                summary = list(csv.DictReader(f))
            self.assertEqual(len(summary), 1)
            self.assertAlmostEqual(float(summary[0]["average_solve_rate"]), 0.5)

    def test_write_outputs_creates_the_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "nested" / "run"
            write_outputs(out, [_rec(5, 1.0)])
            self.assertTrue((out / "results.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
