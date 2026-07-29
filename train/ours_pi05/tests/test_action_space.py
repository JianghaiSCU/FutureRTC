import numpy as np

from ours_pi05.action_space import (
    GRIPPER_DIMS,
    to_delta,
    compute_delta_quantiles,
    qnorm,
    qnorm_inv,
)


def test_to_delta_subtracts_anchor_on_arm_dims_only():
    s_anchor = np.arange(14, dtype=np.float32)
    actions = np.tile(np.arange(14, dtype=np.float32), (3, 1))  # [3, 14], 每步都等于 anchor
    out = to_delta(actions, s_anchor)
    assert out.shape == (3, 14)
    arm = [i for i in range(14) if i not in GRIPPER_DIMS]
    # 手臂维：action == anchor -> delta 为 0
    np.testing.assert_allclose(out[:, arm], 0.0, atol=1e-6)
    # 夹爪维：保持绝对，不减
    np.testing.assert_allclose(out[:, GRIPPER_DIMS[0]], s_anchor[GRIPPER_DIMS[0]])
    np.testing.assert_allclose(out[:, GRIPPER_DIMS[1]], s_anchor[GRIPPER_DIMS[1]])


def test_to_delta_cumsum_is_displacement_from_anchor():
    """delta 表示的意义：cumsum 才是「从陈旧位姿起的累计位移」。"""
    s_anchor = np.zeros(14, dtype=np.float32)
    # 手臂 0 号关节每步前进 0.1（绝对关节角 0.1, 0.2, 0.3）
    actions = np.zeros((3, 14), dtype=np.float32)
    actions[:, 0] = [0.1, 0.2, 0.3]
    out = to_delta(actions, s_anchor)
    np.testing.assert_allclose(out[:, 0], [0.1, 0.2, 0.3], atol=1e-6)


def test_to_delta_batched():
    s_anchor = np.zeros(14, dtype=np.float32)
    actions = np.zeros((2, 5, 14), dtype=np.float32)
    out = to_delta(actions, s_anchor)
    assert out.shape == (2, 5, 14)


def test_qnorm_roundtrip():
    rng = np.random.default_rng(0)
    deltas = rng.normal(size=(1000, 14)).astype(np.float32)
    q = compute_delta_quantiles(deltas)
    assert q["q01"].shape == (14,)
    assert q["q99"].shape == (14,)
    x = qnorm(deltas, q)
    back = qnorm_inv(x, q)
    np.testing.assert_allclose(back, deltas, atol=1e-4)


def test_qnorm_maps_quantile_range_to_unit_interval():
    q = {"q01": np.zeros(14, dtype=np.float32), "q99": np.ones(14, dtype=np.float32)}
    x = qnorm(np.full((1, 14), 0.5, dtype=np.float32), q)
    # [q01, q99] -> [-1, 1]，中点映到 0
    np.testing.assert_allclose(x, 0.0, atol=1e-6)
    # 端点映射：q01 (0.0) -> -1.0，q99 (1.0) -> +1.0
    x_min = qnorm(np.zeros((1, 14), dtype=np.float32), q)
    np.testing.assert_allclose(x_min, -1.0, atol=1e-6)
    x_max = qnorm(np.ones((1, 14), dtype=np.float32), q)
    np.testing.assert_allclose(x_max, 1.0, atol=1e-6)


def test_qnorm_degenerate_dim_does_not_divide_by_zero():
    q = {"q01": np.zeros(14, dtype=np.float32), "q99": np.zeros(14, dtype=np.float32)}
    x = qnorm(np.zeros((1, 14), dtype=np.float32), q)
    assert np.all(np.isfinite(x))
