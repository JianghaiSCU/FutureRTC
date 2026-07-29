"""Delay / execute-horizon sweep enumeration tests (pure, no deps)."""
import pathlib
import sys
import unittest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from motion_prior_handoff.delay_grid import (  # noqa: E402
    build_delay_horizon_pairs,
    limit_horizon_pairs_per_delay,
)


class DelayGridTest(unittest.TestCase):
    def test_pairs_for_chunk_size_8(self):
        pairs = build_delay_horizon_pairs(8, [0, 1, 2, 3, 4])
        # d: execute_horizon in [max(1,d), 8-d]
        self.assertEqual([h for d, h in pairs if d == 0], [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual([h for d, h in pairs if d == 1], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual([h for d, h in pairs if d == 2], [2, 3, 4, 5, 6])
        self.assertEqual([h for d, h in pairs if d == 3], [3, 4, 5])
        self.assertEqual([h for d, h in pairs if d == 4], [4])

    def test_negative_delay_rejected(self):
        with self.assertRaises(ValueError):
            build_delay_horizon_pairs(8, [-1])

    def test_limit_per_delay(self):
        pairs = build_delay_horizon_pairs(8, [0, 1, 2, 3, 4])
        limited = limit_horizon_pairs_per_delay(pairs, 1)
        self.assertEqual(limited, [(0, 1), (1, 1), (2, 2), (3, 3), (4, 4)])

    def test_limit_none_is_identity(self):
        pairs = build_delay_horizon_pairs(8, [1, 2])
        self.assertEqual(limit_horizon_pairs_per_delay(pairs, None), pairs)

    def test_limit_nonpositive_rejected(self):
        with self.assertRaises(ValueError):
            limit_horizon_pairs_per_delay([(1, 1)], 0)


if __name__ == "__main__":
    unittest.main()
