"""真机 cobot 的 action 表示 —— 三处（bank / 训练 / 部署）共用的唯一真相。

为什么必须有这个模块
--------------------
预测器的 ActionTrajectoryEncoder 构造的特征里有 cumsum(a) 与 cumsum(|a|)
（ours_mainline/models/predictor.py:38-39 in build_action_trajectory_features）。这在 LIBERO 的**相对动作**下等于
「累计位移」——正是驱动 latent warp 的运动场。

真机是**绝对关节角**，直接对绝对关节角求 cumsum 毫无意义。所以喂进预测器的
action 必须先转成 delta 表示：手臂维减去锚点 state，夹爪维保持绝对
（这正是 openpi DeltaActions 用的 make_bool_mask(6, -1, 6, -1)，
见 openpi/src/openpi/training/config.py:289）。

锚点 s_anchor 是「陈旧 obs 时刻的 state」，依赖查询时刻，所以**不能**在采集
bank 时固化 —— bank 存绝对 action，delta 在用的时候才算。
"""

from __future__ import annotations

import numpy as np

ACTION_DIM = 14
GRIPPER_DIMS: tuple[int, int] = (6, 13)
ARM_DIMS: tuple[int, ...] = tuple(i for i in range(ACTION_DIM) if i not in GRIPPER_DIMS)


def to_delta(actions: np.ndarray, s_anchor: np.ndarray) -> np.ndarray:
    """绝对关节角 -> delta 表示。

    Args:
        actions: [..., T, 14] 绝对关节角目标
        s_anchor: [14] 锚点 state（陈旧 obs 时刻的关节角）

    Returns:
        [..., T, 14]，手臂维为 action - s_anchor，夹爪维原样保留。
    """
    actions = np.asarray(actions, dtype=np.float32)
    s_anchor = np.asarray(s_anchor, dtype=np.float32)
    if actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"actions last dim must be {ACTION_DIM}, got {actions.shape[-1]}")
    if s_anchor.shape != (ACTION_DIM,):
        raise ValueError(f"s_anchor must be ({ACTION_DIM},), got {s_anchor.shape}")

    out = actions.copy()
    arm = list(ARM_DIMS)
    out[..., arm] = actions[..., arm] - s_anchor[arm]
    return out


def compute_delta_quantiles(deltas: np.ndarray) -> dict[str, np.ndarray]:
    """在 delta 动作上算 q01/q99（用于 qnorm）。deltas: [N, 14]。"""
    deltas = np.asarray(deltas, dtype=np.float32).reshape(-1, ACTION_DIM)
    return {
        "q01": np.quantile(deltas, 0.01, axis=0).astype(np.float32),
        "q99": np.quantile(deltas, 0.99, axis=0).astype(np.float32),
    }


def _scale(quantiles: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    q01 = np.asarray(quantiles["q01"], dtype=np.float32)
    q99 = np.asarray(quantiles["q99"], dtype=np.float32)
    center = (q99 + q01) / 2.0
    half = (q99 - q01) / 2.0
    # 退化维（q01 == q99，例如常量维）不除零
    half = np.where(np.abs(half) < 1e-6, 1.0, half)
    return center, half


def qnorm(deltas: np.ndarray, quantiles: dict[str, np.ndarray]) -> np.ndarray:
    """delta -> [-1, 1]（按 q01/q99 线性映射，不裁剪）。"""
    center, half = _scale(quantiles)
    return ((np.asarray(deltas, dtype=np.float32) - center) / half).astype(np.float32)


def qnorm_inv(x: np.ndarray, quantiles: dict[str, np.ndarray]) -> np.ndarray:
    """qnorm 的逆。"""
    center, half = _scale(quantiles)
    return (np.asarray(x, dtype=np.float32) * half + center).astype(np.float32)
