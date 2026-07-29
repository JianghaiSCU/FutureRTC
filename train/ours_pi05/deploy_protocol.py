"""共享的部署协议契约 —— ``deploy_policy_server_ours_local.py`` 与
``deploy_policy_local_ours_local.py`` 都从这里导入，二者不允许各自重复定义
S/d/H 或切片逻辑：一旦两边各写一份，早晚会漂移出一个静默错误（切片错位、
pad 方向错、committed_actions 对不上 delay），而这类错误在真机上只表现为
"行为不对但不报错"，非常难查。

时序约定（与 sync / naive_async / RTC 三个已验证的真机 client 保持同一把尺子，
以便三方对比）：
    S = 25   每个 window 实际执行的 raw action 步数
    d = 10   推理延迟（launch 到 handoff 之间的 raw action 步数）
    H = 50   server 返回的 chunk 长度（pi0 action_horizon）
    FIRE_STEP = S - d = 15   本 window 执行到第几个 raw action 时发起下一次查询

ours 与 naive_async 唯一的行为差异就在这里：naive_async 因为 obs 是陈旧的，
要扔掉新 chunk 的前 d 步、执行 ``Cn[d:d+S]``；ours 的 server 已经把 obs
预测到了交接时刻，所以整个 chunk 从第 0 步就是"新鲜"的，client 执行
``Cn[0:S]``。这一条是本方法存在的全部意义，混淆了就是在跑一个没有推理延迟
补偿、但自称有补偿的系统。
"""

from __future__ import annotations

import numpy as np

from ours_pi05.action_space import ACTION_DIM, qnorm, to_delta

# MAX_DELAY 在此定义 —— 本模块是共享部署契约的单一来源。ours_pi05.dataset 反过来从这里
# import 它（并 re-export），这样部署时不会为了一个常量而拖入 dataset.py ->
# latent_bank.py -> torch Dataset 一整条训练依赖链。必须与 predictor 的 delay_embedding
# 容量一致（models/predictor.py 的 __init__ 里 max_delay=20）。
MAX_DELAY = 20

# ---------------------------------------------------------------------------
# Timing contract (fixed; matches deploy_policy_local_rtc_local.py /
# deploy_policy_local_rtc_naive_async_local.py / deploy_policy_local_rtc_realtime_chunking_local.py
# so the three methods are comparable on the same clock).
# ---------------------------------------------------------------------------
S: int = 25                     # executed raw actions per window
DELAY_STEPS: int = 10           # inference delay d, in raw action steps
CHUNK_LEN: int = 50             # H: chunk length returned by the server (pi0 action_horizon)
FIRE_STEP: int = S - DELAY_STEPS  # = 15: raw actions executed before firing the next query

METHOD: str = "ours"            # json payload 'method' field the server requires

# Interpolation step sizes: matches the sync (deploy_policy_local_rtc_local.py)
# and RTC (deploy_policy_local_rtc_realtime_chunking_local.py) clients. The
# existing naive-async client uses a flat 0.02 for all 14 dims -- that is a
# confound when comparing the three methods head to head, not a deliberate
# choice, so ours does NOT copy it.
ARM_STEP_SIZES: tuple[float, ...] = (0.015,) * 6 + (0.05,)
INTERP_STEP_SIZES: tuple[float, ...] = ARM_STEP_SIZES + ARM_STEP_SIZES  # both arms, 14 dims


def executed_slice(chunk: np.ndarray, s: int = S) -> np.ndarray:
    """The slice of a freshly-returned chunk the client actually executes.

    ours executes ``C_n[0:S]`` -- NOT ``C_n[d:d+S]`` (that is the naive_async
    rule). The server has already predicted the handoff-time observation, so
    the whole chunk is "fresh" starting at index 0; there is nothing stale to
    discard.
    """
    chunk = np.asarray(chunk)
    if chunk.shape[0] < s:
        raise ValueError(f"chunk has {chunk.shape[0]} rows, need >= {s} to slice [0:{s}]")
    return chunk[0:s]


def committed_actions_from_slice(
    action_slice: np.ndarray,
    *,
    fire_step: int = FIRE_STEP,
    delay: int = DELAY_STEPS,
) -> np.ndarray:
    """``committed_actions`` sent with the query fired at ``fire_step``.

    The query is fired at raw-action index ``fire_step`` of the CURRENT window
    (``t = e - d``); the ``delay`` raw actions of *this* window that are
    guaranteed to execute between the fire instant and the handoff instant
    ``e`` are exactly ``action_slice[fire_step : fire_step + delay]``
    (= ``action_slice[15:25]`` for the fixed S=25/d=10 contract). These are the
    only actions the server can treat as "certainly executed" when it
    forward-predicts the handoff-time observation.
    """
    action_slice = np.asarray(action_slice)
    committed = action_slice[fire_step : fire_step + delay]
    if committed.shape != (delay, ACTION_DIM):
        raise ValueError(
            f"committed_actions must be shape ({delay}, {ACTION_DIM}), got {committed.shape} "
            f"(action_slice had {action_slice.shape[0]} rows, fire_step={fire_step})"
        )
    return committed


def build_padded_motion(
    committed_actions: np.ndarray,
    s_anchor: np.ndarray,
    action_quantiles: dict,
    delay: int,
    *,
    max_delay: int = MAX_DELAY,
) -> np.ndarray:
    """预测器 ``motion_actions`` 输入的精确构造，逐字对齐
    ``PerFrameLatentDataset.__getitem__`` (ours_pi05/dataset.py:136-151) 与
    ``FastBatchLoader._fill_one`` (ours_pi05/fast_loader.py:113-127)：

        delta  = to_delta(committed_actions, s_anchor)      # 手臂维减锚点，夹爪维原样
        motion = qnorm(delta, action_quantiles)             # [-1, 1]
        padded = zeros((MAX_DELAY, 14)); padded[:d] = motion  # **左对齐**

    左对齐是强制的：predictor 的 ``valid_mask = step_ids < delay`` 与
    ``endpoint = x[:, delay - 1]``（ours_pi05/models/predictor.py:200/56）都假设
    有效步在最前面。右对齐会让模型在"有效"区间里读到全零 —— 不报错，但
    cumsum/path_magnitude 等特征全部作废，训练与部署的分布也对不上。

    Args:
        committed_actions: [delay, 14] 绝对关节角，必须恰好 ``delay`` 行（不是
            ``>= delay`` —— 调用方必须先按 delay 精确切好，这里不做静默截断，
            免得 shape 不对时吞掉一个本该在上游就暴露的协议错误）。
        s_anchor: [14] 陈旧 obs 时刻（查询发出时刻）的 state，即 ``to_delta``
            的锚点。**不是** 交接时刻的校正态 —— 那是喂给 predictor 的
            ``state`` 参数，是另一个量，不要在这里传错。
        delay: 提前量，单位 step。
        max_delay: predictor 的 delay_embedding 容量，默认取
            ``ours_pi05.dataset.MAX_DELAY`` (=20)。

    Returns:
        [max_delay, 14] float32，左对齐补零。
    """
    committed_actions = np.asarray(committed_actions, dtype=np.float32)
    delay = int(delay)
    if delay < 1 or delay > max_delay:
        raise ValueError(f"delay must be in [1, {max_delay}], got {delay}")
    if committed_actions.shape != (delay, ACTION_DIM):
        raise ValueError(
            f"committed_actions must be shape ({delay}, {ACTION_DIM}) to match delay={delay}, "
            f"got {committed_actions.shape}"
        )

    delta = to_delta(committed_actions, s_anchor)
    motion = qnorm(delta, action_quantiles)

    padded = np.zeros((max_delay, ACTION_DIM), dtype=np.float32)
    padded[:delay] = motion
    return padded


def gap_log_filename(task: str, timestamp: str) -> str:
    """``outputs/gap_log_<task>_<timestamp>.npz`` 的文件名（不含目录）。"""
    return f"gap_log_{task}_{timestamp}.npz"


def save_gap_log(path, committed: np.ndarray, measured: np.ndarray) -> None:
    """落盘 (下发的最后一个关节目标, 交接时刻实测编码器值) 配对。

    Args:
        committed: [N, 14] -- a_{e-1}，每个 window 下发的最后一个 committed action。
        measured:  [N, 14] -- s_meas(e)，交接瞬间编码器实测的关节角。
    """
    committed = np.asarray(committed, dtype=np.float32)
    measured = np.asarray(measured, dtype=np.float32)
    if committed.shape != measured.shape:
        raise ValueError(f"committed/measured shape mismatch: {committed.shape} vs {measured.shape}")
    if committed.ndim != 2 or committed.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected [N, {ACTION_DIM}], got {committed.shape}")
    np.savez(path, committed=committed, measured=measured)
