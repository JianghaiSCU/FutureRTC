"""交接时刻 state 的预测器。

在真机上这是**恒等映射**，不是近似。

原因：数据集里 action 就是下一帧实测到的从臂关节角。原始 HDF5 实测
（/root/zyx/data/cvpr2026_RTC/plates_stacking/episode_0.hdf5）：

    max|qpos_(t+1) - action_t| = 0.0        <- 精确，float32 零误差
    max|qpos_(t)   - action_t| = 0.175

所以 s_{t+d} ≡ a_{t+d-1}，物理 proxy 就是「最后一个已提交的 action」。

LIBERO 版需要两级（robosuite OSC 前向积分 + 残差 MLP），是因为那边是**相对动作**，
action -> state 做不到恒等。真机是绝对关节角，两级双双退化。

残差 MLP 的接口保留（`residual_ckpt`）但默认关闭：数据集里残差 label 恒为 0，
训不出东西；真实执行误差（编码器实测 vs 下发目标）只在部署时存在，采集数据里
不可见。真机客户端会记录这一对，若量出显著 gap，再训残差接进来。
"""

from __future__ import annotations

import numpy as np

from ours_pi05.action_space import ACTION_DIM


class Corrector:
    def __init__(self, residual_ckpt: str | None = None):
        if residual_ckpt is not None:
            raise NotImplementedError(
                "残差 MLP 尚未训练。数据集里 label 恒为 0；需要先从真机日志采集 "
                "(下发目标, 实测编码器) 配对数据。见 spec §4.4。"
            )
        self.residual = None

    def __call__(
        self,
        base_state: np.ndarray,
        committed_actions: np.ndarray,
        delay: int,
    ) -> np.ndarray:
        """预测 t + delay 时刻的 state。

        Args:
            base_state: [14] 陈旧 obs 时刻的关节角
            committed_actions: [>=delay, 14] 已提交、必然执行的绝对关节角目标
            delay: 提前量，单位 step

        Returns:
            [14] 交接时刻的关节角
        """
        base_state = np.asarray(base_state, dtype=np.float32).reshape(-1)
        if base_state.shape != (ACTION_DIM,):
            raise ValueError(f"base_state must be ({ACTION_DIM},), got {base_state.shape}")
        delay = int(delay)
        if delay < 0:
            raise ValueError(f"delay must be >= 0, got {delay}")
        if delay == 0:
            return base_state.copy()

        # Validate committed_actions shape before reshaping
        committed_arr = np.asarray(committed_actions, dtype=np.float32)
        if committed_arr.ndim != 2 or committed_arr.shape[1] != ACTION_DIM:
            raise ValueError(
                f"committed_actions must be shape [>=delay, {ACTION_DIM}], "
                f"got ndim={committed_arr.ndim} shape={committed_arr.shape}"
            )
        committed = committed_arr

        if committed.shape[0] < delay:
            raise ValueError(f"need >= {delay} committed actions, got {committed.shape[0]}")

        return committed[delay - 1].copy()
