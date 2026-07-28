import json
import pathlib
import tempfile
import unittest

from eval.aggregate import four_suite_average, load_records, suite_delay_table

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def _rows(values):
    out = []
    for (suite, delay), rate in values.items():
        for i in range(2):
            out.append({"benchmark": suite, "base_model": "pi05", "method": "predictor_corrector",
                        "delay": delay, "task_or_level_id": "0", "episode_idx": i,
                        "solve_rate": rate, "n_trials": 1, "execution_time": 1.0, "seed": 7})
    return out


class AggregateTest(unittest.TestCase):
    def test_table_groups_by_delay_and_suite(self):
        table = suite_delay_table(_rows({("libero_spatial", 5): 1.0, ("libero_object", 5): 0.0,
                                         ("libero_spatial", 10): 0.5}))
        self.assertEqual(sorted(table), [5, 10])
        self.assertAlmostEqual(table[5]["libero_spatial"], 1.0)
        self.assertAlmostEqual(table[5]["libero_object"], 0.0)

    def test_average_over_four_suites(self):
        table = suite_delay_table(_rows({(s, 5): v for s, v in zip(SUITES, [1.0, 0.8, 0.6, 0.4])}))
        self.assertAlmostEqual(four_suite_average(table)[5], 0.7)

    def test_average_is_none_when_a_suite_is_missing(self):
        table = suite_delay_table(_rows({(s, 5): 1.0 for s in SUITES[:3]}))
        self.assertIsNone(four_suite_average(table)[5])

    def test_solve_rates_average_across_episodes(self):
        rows = _rows({("libero_spatial", 5): 1.0})
        rows[1]["solve_rate"] = 0.0
        self.assertAlmostEqual(suite_delay_table(rows)[5]["libero_spatial"], 0.5)

    def test_load_records_reads_a_directory_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for suite in ("libero_spatial", "libero_object"):
                d = root / f"{suite}_d5"
                d.mkdir(parents=True)
                with (d / "results.jsonl").open("w") as f:
                    for row in _rows({(suite, 5): 1.0}):
                        f.write(json.dumps(row) + "\n")
            self.assertEqual(len(load_records([root])), 4)

    def test_load_records_accepts_a_direct_jsonl_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "results.jsonl"
            with path.open("w") as f:
                for row in _rows({("libero_goal", 15): 1.0}):
                    f.write(json.dumps(row) + "\n")
            self.assertEqual(len(load_records([path])), 2)

    def test_malformed_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "results.jsonl"
            path.write_text(
                '{"benchmark": "libero_goal", "delay": 5, "solve_rate": 1.0}\nnot json\n')
            self.assertEqual(len(load_records([path])), 1)


if __name__ == "__main__":
    unittest.main()
