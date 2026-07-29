import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd

SCRIPT_PATH = pathlib.Path(__file__).resolve().parent / "subsample_lerobot_dataset.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("subsample_lerobot_dataset", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stride_indices_keep_every_nth_frame():
    module = _load_script_module()

    assert module.stride_indices(10, 4) == [0, 4, 8]
    assert module.stride_indices(4, 4) == [0]


def test_subsample_frame_table_rewrites_lerobot_indices():
    module = _load_script_module()
    episode_table = pd.DataFrame(
        {
            "timestamp": np.arange(10, dtype=np.float32) / 30.0,
            "frame_index": np.arange(10),
            "episode_index": np.zeros(10, dtype=np.int64),
            "index": np.arange(100, 110),
            "task_index": np.zeros(10, dtype=np.int64),
            "action.effector.position": [np.array([i, i + 1], dtype=np.float32) for i in range(10)],
        }
    )

    out = module.subsample_frame_table(episode_table, stride=4, episode_index=3, global_start_index=20, fps=7.5)

    assert len(out) == 3
    assert out["frame_index"].tolist() == [0, 1, 2]
    assert out["episode_index"].tolist() == [3, 3, 3]
    assert out["index"].tolist() == [20, 21, 22]
    np.testing.assert_allclose(out["timestamp"].to_numpy(), np.array([0.0, 1.0 / 7.5, 2.0 / 7.5]))
    np.testing.assert_array_equal(np.stack(out["action.effector.position"]), np.array([[0, 1], [4, 5], [8, 9]]))


def test_update_info_keeps_structure_and_updates_counts():
    module = _load_script_module()
    info = {
        "fps": 30,
        "total_frames": 10,
        "total_episodes": 2,
        "features": {"a": {"dtype": "float32"}},
    }

    updated = module.update_info(info, fps=7.5, episode_lengths=[3, 2])

    assert updated["fps"] == 7.5
    assert updated["total_frames"] == 5
    assert updated["total_episodes"] == 2
    assert updated["features"] == info["features"]


def test_episode_stats_counts_match_subsampled_rows():
    module = _load_script_module()
    episode_table = pd.DataFrame(
        {
            "observation.state.effector.position": [
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([2.0, 3.0], dtype=np.float32),
            ],
            "timestamp": np.array([0.0, 0.5], dtype=np.float32),
            "frame_index": np.array([0, 1], dtype=np.int64),
            "episode_index": np.array([4, 4], dtype=np.int64),
            "index": np.array([10, 11], dtype=np.int64),
            "task_index": np.array([0, 0], dtype=np.int64),
        }
    )

    stats = module.compute_episode_stats(4, episode_table)

    assert stats["episode_index"] == 4
    assert stats["stats"]["observation.state.effector.position"]["count"] == [2]
    assert stats["stats"]["observation.state.effector.position"]["min"] == [0.0, 1.0]
    assert stats["stats"]["observation.state.effector.position"]["max"] == [2.0, 3.0]
    json.dumps(stats)
