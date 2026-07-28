import unittest

from eval.schedule import (
    committed_slice, compute_delay_batch_layout, query_step, stale_step, step_index, step_window,
)


class ScheduleTest(unittest.TestCase):
    def test_window_zero_executes_the_fresh_chunk(self):
        self.assertEqual([step_index(t, 10) for t in range(10)], list(range(10)))

    def test_every_window_executes_from_index_zero(self):
        for m in range(4):
            self.assertEqual([step_index(m * 10 + off, 10) for off in range(10)], list(range(10)))

    def test_step_window(self):
        self.assertEqual([step_window(t, 5) for t in (0, 4, 5, 9, 10)], [0, 0, 1, 1, 2])

    def test_query_step(self):
        self.assertEqual([query_step(m, 25) for m in range(4)], [0, 25, 50, 75])

    def test_stale_step_is_the_query_minus_the_delay(self):
        self.assertEqual(stale_step(2, 25, 10), 40)
        self.assertEqual(stale_step(1, 25, 0), 25)

    def test_committed_slice_length_equals_the_delay(self):
        lo, hi = committed_slice(25, 10)
        self.assertEqual((lo, hi), (15, 25))
        self.assertEqual(hi - lo, 10)

    def test_committed_slice_is_empty_at_zero_delay(self):
        lo, hi = committed_slice(25, 0)
        self.assertEqual(hi - lo, 0)

    def test_delay_above_stride_rejected(self):
        with self.assertRaises(ValueError):
            committed_slice(10, 11)

    def test_bad_arguments_rejected(self):
        for call in (lambda: step_index(-1, 10), lambda: step_index(0, 0),
                     lambda: query_step(-1, 10), lambda: stale_step(1, 10, -1)):
            with self.assertRaises(ValueError):
                call()

    def test_batch_layout_divides_the_cpu_budget(self):
        self.assertEqual(compute_delay_batch_layout(4, 50, 128, cpu_fraction=0.32), 10)
        self.assertEqual(compute_delay_batch_layout(3, 50, 128, cpu_fraction=0.24), 10)

    def test_batch_layout_never_exceeds_the_trial_count(self):
        self.assertEqual(compute_delay_batch_layout(1, 5, 128, cpu_fraction=0.8), 5)

    def test_batch_layout_raises_when_the_budget_is_too_small(self):
        with self.assertRaises(ValueError):
            compute_delay_batch_layout(8, 50, 8, cpu_fraction=0.1)


if __name__ == "__main__":
    unittest.main()
