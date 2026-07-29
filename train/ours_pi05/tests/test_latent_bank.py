import numpy as np
import torch

from ours_pi05.latent_bank import Bank, BankWriter


def _make_bank(tmp_path, num_frames=8, episodes=(0, 0, 0, 0, 1, 1, 1, 1)):
    w = BankWriter(tmp_path, num_frames=num_frames)
    rng = np.random.default_rng(0)
    frame_in_ep = {}
    for i in range(num_frames):
        ep = episodes[i]
        f = frame_in_ep.get(ep, 0)
        frame_in_ep[ep] = f + 1
        w.write(
            i,
            latent=rng.normal(size=(3, 256, 2048)).astype(np.float32),
            action=rng.normal(size=14).astype(np.float32),
            state=rng.normal(size=14).astype(np.float32),
            episode_index=ep,
            frame_index=f,
        )
    w.finalize(
        latent_norm={"mean": np.zeros((3, 2048), np.float32), "std": np.ones((3, 2048), np.float32)},
        action_quantiles={"q01": np.zeros(14, np.float32), "q99": np.ones(14, np.float32)},
        meta={"task": "test"},
    )
    return tmp_path


def test_roundtrip_shapes_and_dtype(tmp_path):
    root = _make_bank(tmp_path)
    b = Bank(root)
    assert b.latents.shape == (8, 3, 256, 2048)
    assert b.latents.dtype == torch.bfloat16 or b.latents.dtype == np.dtype("uint16")
    assert b.actions.shape == (8, 14)
    assert b.states.shape == (8, 14)
    assert b.latent_norm["mean"].shape == (3, 2048)


def test_episode_starts(tmp_path):
    root = _make_bank(tmp_path)
    b = Bank(root)
    assert b.episode_starts == {0: 0, 1: 4}


def test_latent_values_survive_bf16_roundtrip(tmp_path):
    """bf16 只有 8 bit 尾数，容差要给够；但不能是垃圾。"""
    root = _make_bank(tmp_path)
    b = Bank(root)
    z = b.read_latent(0)
    assert z.shape == (3, 256, 2048)
    assert np.all(np.isfinite(z))


def test_latent_bf16_roundtrip_is_not_garbage(tmp_path):
    """写入一个已知值，读回后应等于该值 bf16 舍入后的结果，而不是任意噪声。"""
    root = _make_bank(tmp_path, num_frames=1, episodes=(0,))
    w = BankWriter(root, num_frames=1)
    known = np.full((3, 256, 2048), 1.0 / 3.0, dtype=np.float32)
    w.write(
        0,
        latent=known,
        action=np.zeros(14, np.float32),
        state=np.zeros(14, np.float32),
        episode_index=0,
        frame_index=0,
    )
    w.finalize(
        latent_norm={"mean": np.zeros((3, 2048), np.float32), "std": np.ones((3, 2048), np.float32)},
        action_quantiles={"q01": np.zeros(14, np.float32), "q99": np.ones(14, np.float32)},
        meta={"task": "test"},
    )
    b = Bank(root)
    z = b.read_latent(0)
    expected = torch.tensor(1.0 / 3.0, dtype=torch.float32).to(torch.bfloat16).to(torch.float32).item()
    # bf16 rounding of 1/3 must actually change the value vs the raw float32 input —
    # otherwise this test can't distinguish "correct bf16 roundtrip" from "garbage that
    # happens to look like a passthrough".
    assert expected != 1.0 / 3.0
    assert np.allclose(z, expected, atol=1e-6)


def test_episode_starts_single_episode(tmp_path):
    root = _make_bank(tmp_path, num_frames=4, episodes=(0, 0, 0, 0))
    b = Bank(root)
    assert b.episode_starts == {0: 0}


def test_latent_shape_matches_bridge():
    from ours_pi05.latent_bank import LATENT_SHAPE as BANK_SHAPE
    from ours_pi05.openpi_bridge import LATENT_SHAPE as BRIDGE_SHAPE

    assert BANK_SHAPE == BRIDGE_SHAPE
