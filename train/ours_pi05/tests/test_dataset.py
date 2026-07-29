import numpy as np
import pytest
import torch

from ours_pi05.dataset import PerFrameLatentDataset
from ours_pi05.latent_bank import Bank, BankWriter
from ours_pi05.models.corrector import Corrector


@pytest.fixture
def bank(tmp_path):
    # 2 个 episode，各 20 帧
    n = 40
    w = BankWriter(tmp_path, num_frames=n)
    rng = np.random.default_rng(0)
    for i in range(n):
        ep = 0 if i < 20 else 1
        w.write(
            i,
            latent=rng.normal(size=(3, 256, 2048)).astype(np.float32),
            action=rng.normal(size=14).astype(np.float32),
            state=rng.normal(size=14).astype(np.float32),
            episode_index=ep,
            frame_index=i % 20,
        )
    w.finalize(
        latent_norm={"mean": np.zeros((3, 2048), np.float32), "std": np.ones((3, 2048), np.float32)},
        action_quantiles={"q01": -np.ones(14, np.float32), "q99": np.ones(14, np.float32)},
        meta={"task": "test"},
    )
    return Bank(tmp_path)


def test_item_shapes(bank):
    ds = PerFrameLatentDataset(bank, delay_set=[10])
    item = ds[0]
    assert item["z"].shape == (3, 256, 2048)
    assert item["z_target"].shape == (3, 256, 2048)
    assert item["z_init"].shape == (3, 256, 2048)
    assert item["motion_actions"].shape == (20, 14)  # 补零到 max_delay=20（左对齐）
    assert item["state"].shape == (14,)
    assert int(item["delay"]) == 10


def test_no_sample_crosses_episode_boundary(bank):
    """episode 0 是 0..19。d=10 时锚点最多到 9（因为 target 是 t+10 <= 19）。"""
    ds = PerFrameLatentDataset(bank, delay_set=[10])
    anchors = ds.anchors
    for row in anchors:
        assert bank.episode_index[row] == bank.episode_index[row + 10]
    # 每个 episode 20 帧、d=10 -> 各 10 个合法锚点
    assert len(anchors) == 20


def test_z_init_is_episode_first_frame(bank):
    ds = PerFrameLatentDataset(bank, delay_set=[10])
    item = ds[0]
    row = ds.anchors[0]
    ep = int(bank.episode_index[row])
    expected = bank.read_latent(bank.episode_starts[ep])
    got = item["z_init"].numpy() * bank.latent_norm["std"][:, None, :] + bank.latent_norm["mean"][:, None, :]
    np.testing.assert_allclose(got, expected, atol=0.3)  # bf16 精度


def test_motion_actions_are_qnorm_delta_left_aligned(bank):
    """**左对齐**：前 d 行是 qnorm 的 delta，其余补零。

    这必须与 predictor 的 build_action_trajectory_features 对齐 ——
    它用 valid_mask = step_ids < delay（ours_mainline/models/predictor.py:34），
    且 endpoint 取 x[:, delay - 1]（:184）。搞成右对齐会静默毁掉整个训练：
    模型会读到全零的「有效」步，cumsum 特征全废。
    """
    ds = PerFrameLatentDataset(bank, delay_set=[10])
    item = ds[0]
    ma = item["motion_actions"].numpy()
    assert not np.allclose(ma[:10], 0.0)
    np.testing.assert_allclose(ma[10:], 0.0, atol=0)


def test_state_is_corrected_handoff_state(bank):
    """state 喂给预测器的是交接时刻（t+d）的**校正后** state，不是陈旧的 s_anchor。

    训练和部署必须过同一个 Corrector（见 ours_mainline/README.md:80，
    以及参考 checkpoint 训练元数据里的 state_source=corrector）——
    否则两边条件的分布不一致，模型会静默学错。
    在这条真机数据上 Corrector 是恒等映射：返回 committed_actions[d-1]，
    因为采集时的 action_t 本来就是下一帧实测关节角 qpos_{t+1}。
    """
    ds = PerFrameLatentDataset(bank, delay_set=[10])
    item = ds[0]
    row = ds.anchors[0]
    d = int(item["delay"])
    expected = Corrector()(bank.states[row], bank.actions[row : row + d], d)
    np.testing.assert_allclose(item["state"].numpy(), expected, atol=1e-6)
    # 恒等退化：等于最后一个已提交 action。
    np.testing.assert_allclose(item["state"].numpy(), bank.actions[row + d - 1], atol=1e-6)


def test_state_is_not_stale_anchor_state(tmp_path):
    """回归防护：state 不能悄悄退回陈旧的 s_anchor = bank.states[row]。

    构造一个合成 bank，让 states[row]（陈旧）与 actions[row+d-1]（校正后的交接
    state，Corrector 在本数据上恒等于 committed_actions[d-1]）量级悬殊，
    确认返回的 state 是后者而非前者。没有这条测试，一次把 state 改回
    `bank.states[row]` 的回归会静默通过所有其他断言。
    """
    n = 40
    w = BankWriter(tmp_path, num_frames=n)
    rng = np.random.default_rng(1)
    for i in range(n):
        ep = 0 if i < 20 else 1
        # state 和 action 用不同量级，确保 states[row] 与 actions[row+d-1]（==
        # 校正后的交接 state）差异悬殊。
        w.write(
            i,
            latent=rng.normal(size=(3, 256, 2048)).astype(np.float32),
            action=(rng.normal(size=14).astype(np.float32) + 100.0),
            state=rng.normal(size=14).astype(np.float32),
            episode_index=ep,
            frame_index=i % 20,
        )
    w.finalize(
        latent_norm={"mean": np.zeros((3, 2048), np.float32), "std": np.ones((3, 2048), np.float32)},
        action_quantiles={"q01": -np.ones(14, np.float32) * 200, "q99": np.ones(14, np.float32) * 200},
        meta={"task": "test"},
    )
    distinct_bank = Bank(tmp_path)

    ds = PerFrameLatentDataset(distinct_bank, delay_set=[10])
    item = ds[0]
    row = ds.anchors[0]

    stale = distinct_bank.states[row]
    got = item["state"].numpy()
    assert not np.allclose(got, stale, atol=1e-3)
    # 正向确认：确实等于校正后的交接 state。
    d = int(item["delay"])
    np.testing.assert_allclose(got, distinct_bank.actions[row + d - 1], atol=1e-6)


def test_deterministic_with_seed(bank):
    a = PerFrameLatentDataset(bank, delay_set=[5, 10], seed=7)[3]
    b = PerFrameLatentDataset(bank, delay_set=[5, 10], seed=7)[3]
    assert int(a["delay"]) == int(b["delay"])
