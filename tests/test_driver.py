import unittest

import numpy as np

from eval.driver import assemble_action, slice_raw_obs


class SliceRawObsTest(unittest.TestCase):
    def test_slices_arrays_along_the_env_axis(self):
        obs = {"pixels": np.arange(40).reshape(8, 5), "state": np.arange(8)}
        out = slice_raw_obs(obs, 2, 5)
        self.assertEqual(out["pixels"].shape, (3, 5))
        self.assertEqual(out["state"].tolist(), [2, 3, 4])

    def test_recurses_into_nested_dicts(self):
        obs = {"images": {"cam0": np.arange(8), "cam1": np.arange(8) * 2}}
        out = slice_raw_obs(obs, 1, 3)
        self.assertEqual(out["images"]["cam0"].tolist(), [1, 2])
        self.assertEqual(out["images"]["cam1"].tolist(), [2, 4])

    def test_slices_lists_and_preserves_type(self):
        out = slice_raw_obs({"task": ["a", "b", "c", "d"]}, 1, 3)
        self.assertEqual(out["task"], ["b", "c"])

    def test_passes_scalars_and_strings_through(self):
        out = slice_raw_obs({"n": 5, "name": "libero_spatial"}, 0, 2)
        self.assertEqual(out["n"], 5)
        self.assertEqual(out["name"], "libero_spatial")


class AssembleActionTest(unittest.TestCase):
    def setUp(self):
        self.b_t, self.s, self.A = 2, 5, 7
        self.delays = [5, 10]
        # chunk value encodes (chunk index * 10 + group) so an assembled column is identifiable
        self.chunks = {
            g: {m: np.repeat((np.arange(50) * 10 + g)[None, :, None], self.b_t, axis=0)
                   .repeat(self.A, axis=2).astype(np.float64)
                for m in range(4)}
            for g in range(2)
        }

    def test_output_shape(self):
        out = assemble_action(0, self.s, self.delays, self.chunks, self.b_t)
        self.assertEqual(out.shape, (self.b_t * len(self.delays), 1, self.A))

    def test_each_group_executes_from_index_zero_in_every_window(self):
        for m in range(3):
            for off in range(self.s):
                out = assemble_action(m * self.s + off, self.s, self.delays, self.chunks, self.b_t)
                self.assertEqual(float(out[0, 0, 0]), off * 10 + 0)          # group 0
                self.assertEqual(float(out[self.b_t, 0, 0]), off * 10 + 1)   # group 1

    def test_groups_are_stacked_in_delay_order(self):
        out = assemble_action(3, self.s, self.delays, self.chunks, self.b_t)
        self.assertTrue(np.all(out[: self.b_t, 0, 0] == 30))
        self.assertTrue(np.all(out[self.b_t:, 0, 0] == 31))

    def test_index_is_clamped_to_the_chunk_length(self):
        short = {g: {0: np.zeros((self.b_t, 3, self.A)) + np.arange(3)[None, :, None]}
                 for g in range(2)}
        out = assemble_action(4, self.s, self.delays, short, self.b_t)
        self.assertTrue(np.all(out[:, 0, 0] == 2))  # clamped to the last available index



class LerobotRootResolutionTest(unittest.TestCase):
    """`--lerobot-root` defaults to None when LeRobot is installed; nothing may stringify that.

    Regression: `pathlib.Path(None)` raised, and `str(None)` reached the spawn workers as the
    literal path "None". Both only surface once a real env is built, so they are pinned here.
    """

    def test_append_lerobot_src_accepts_none(self):
        import importlib.util
        from eval.async_env import _append_lerobot_src
        if importlib.util.find_spec("lerobot") is None:
            self.skipTest("lerobot not installed; None would legitimately raise")
        _append_lerobot_src(None)  # must not raise

    def test_env_fns_do_not_stringify_a_none_root(self):
        from eval.async_env import make_spawn_safe_libero_env_fns
        fns = make_spawn_safe_libero_env_fns(
            lerobot_root=None, suite_name="libero_spatial", task_id=0, n_envs=2,
            camera_name="agentview", init_states=True, episode_length=220,
            control_mode="relative", gym_kwargs={})
        for fn in fns:
            self.assertIsNone(fn.keywords["lerobot_root"])

    def test_env_fns_preserve_an_explicit_root(self):
        from eval.async_env import make_spawn_safe_libero_env_fns
        fns = make_spawn_safe_libero_env_fns(
            lerobot_root="/some/checkout", suite_name="libero_spatial", task_id=0, n_envs=1,
            camera_name="agentview", init_states=True, episode_length=220,
            control_mode="relative", gym_kwargs={})
        self.assertEqual(fns[0].keywords["lerobot_root"], "/some/checkout")

if __name__ == "__main__":
    unittest.main()
