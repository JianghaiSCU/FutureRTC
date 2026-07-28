"""SO(3) controller-target proxy and the full-state residual feature/apply pipeline.

Vendored from ``action_state_residual_repro/scripts/lerobot_action_state_residual.py`` (the
action-to-state residual repro bundle) so this release carries no external path dependency. Only
the full-state residual path is carried over; the position-only residual variant is omitted.

The corrector estimates the proprioceptive state at the handoff from the measured state ``d`` steps
earlier plus the ``d`` actions committed in between:

1. forward-integrate those actions through the LIBERO OSC controller-target approximation to get a
   proxy state (``make_controller_target_state_canonical_prev``);
2. build a 177-dim feature vector from (base state, right-aligned actions, action mask, proxy,
   normalized delay);
3. an MLP predicts the normalized residual between proxy and truth;
4. denormalize it and compose it onto the proxy, with SO(3) composition on the rotation block.

All functions are numpy and batched on a leading ``B`` axis, EXCEPT ``axisangle_to_matrix`` /
``matrix_to_axisangle`` / ``canonicalize_axisangle_to_reference``, which act on a single 3-vector.

ACTION SPACE: every action argument here is ENV space (physical OSC deltas). Converting from the
latent bank's quantile-normalized space is the caller's job -- see ``common.action_space``.
"""
from __future__ import annotations

import numpy as np

MAX_DELAY = 20
STATE_DIM = 8
FEATURE_DIM = 177  # base(8) + actions(20*7) + mask(20) + proxy(8) + delay(1)

OSC_POSE_POSITION_SCALE = np.asarray([0.05, 0.05, 0.05], dtype=np.float64)
OSC_POSE_ORIENTATION_SCALE = np.asarray([0.5, 0.5, 0.5], dtype=np.float64)

PANDA_GRIPPER_QPOS_BIAS = np.asarray([0.02, -0.02], dtype=np.float64)
PANDA_GRIPPER_QPOS_WEIGHT = np.asarray([0.02, 0.02], dtype=np.float64)
PANDA_GRIPPER_DIRECTION = np.asarray([-1.0, 1.0], dtype=np.float64)
PANDA_GRIPPER_SPEED = 0.01


def _as_batched_state(state: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(state)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim < 2 or value.shape[-1] != 8:
        raise ValueError(f"{name} must have shape (..., 8), got {value.shape}.")
    return value


def _as_batched_actions(actions: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(actions)
    if value.ndim == 2:
        value = value[None, :, :]
    if value.ndim < 3 or value.shape[-1] != 7:
        raise ValueError(f"{name} must have shape (..., T, 7), got {value.shape}.")
    return value


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def axisangle_to_matrix(axisangle: np.ndarray) -> np.ndarray:
    """Convert a 3D axis-angle vector to a rotation matrix."""

    vector = np.asarray(axisangle, dtype=np.float64)
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / theta
    k = _skew(axis)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def matrix_to_axisangle(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a 3D axis-angle vector."""

    mat = np.asarray(matrix, dtype=np.float64)
    cos_theta = float(np.clip((np.trace(mat) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)

    sin_theta = float(np.sin(theta))
    if abs(sin_theta) > 1e-8:
        axis = np.asarray(
            [
                mat[2, 1] - mat[1, 2],
                mat[0, 2] - mat[2, 0],
                mat[1, 0] - mat[0, 1],
            ],
            dtype=np.float64,
        ) / (2.0 * sin_theta)
        return axis * theta

    # Near pi the standard formula is unstable; recover the dominant axis from
    # the diagonal and use off-diagonal signs for continuity.
    axis = np.sqrt(np.maximum(np.diag(mat) + 1.0, 0.0) / 2.0)
    if mat[2, 1] - mat[1, 2] < 0:
        axis[0] = -axis[0]
    if mat[0, 2] - mat[2, 0] < 0:
        axis[1] = -axis[1]
    if mat[1, 0] - mat[0, 1] < 0:
        axis[2] = -axis[2]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = axis / norm
    return axis * theta


def canonicalize_axisangle_to_reference(axisangle: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Choose an equivalent axis-angle branch nearest to ``reference``."""

    axisangle = np.asarray(axisangle, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    theta = float(np.linalg.norm(axisangle))
    if theta < 1e-12:
        return axisangle.copy()
    axis = axisangle / theta
    candidates = [axisangle + (2.0 * np.pi * k) * axis for k in range(-2, 3)]
    return min(candidates, key=lambda candidate: float(np.linalg.norm(candidate - reference))).copy()


def make_controller_target_state_canonical_prev(previous_state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Convert known actions into an 8D LIBERO state proxy.

    ``actions`` may have length zero. In that case this returns a copy of
    ``previous_state``, which is the d=0 oracle-GT boundary behavior.
    """

    previous_state = _as_batched_state(previous_state, "previous_state")
    actions = _as_batched_actions(actions, "actions")
    if previous_state.shape[:-1] != actions.shape[:-2]:
        raise ValueError(
            "previous_state batch dimensions must match actions batch dimensions: "
            f"{previous_state.shape[:-1]} != {actions.shape[:-2]}"
        )
    if actions.shape[-2] == 0:
        return np.array(previous_state, dtype=np.result_type(previous_state, np.float32), copy=True)

    out = np.empty(previous_state.shape, dtype=np.result_type(previous_state, actions, np.float32))
    flat_previous = np.asarray(previous_state, dtype=np.float64).reshape(-1, 8)
    flat_actions = np.asarray(actions, dtype=np.float64).reshape(-1, actions.shape[-2], 7)
    flat_out = out.reshape(-1, 8)

    for index, (state, action_seq) in enumerate(zip(flat_previous, flat_actions, strict=True)):
        pos = state[:3].copy()
        ori_mat = axisangle_to_matrix(state[3:6])
        ori_ref = state[3:6].copy()
        gripper_normalized = np.clip(
            (state[6:8] - PANDA_GRIPPER_QPOS_BIAS) / PANDA_GRIPPER_QPOS_WEIGHT,
            -1.0,
            1.0,
        )

        for action in action_seq:
            arm = np.clip(action[:6], -1.0, 1.0)
            pos = pos + arm[:3] * OSC_POSE_POSITION_SCALE

            delta_axisangle = arm[3:6] * OSC_POSE_ORIENTATION_SCALE
            if not np.allclose(delta_axisangle, 0.0):
                ori_mat = axisangle_to_matrix(delta_axisangle) @ ori_mat
            raw_axisangle = matrix_to_axisangle(ori_mat)
            ori_ref = canonicalize_axisangle_to_reference(raw_axisangle, ori_ref)

            gripper_normalized = np.clip(
                gripper_normalized + PANDA_GRIPPER_DIRECTION * PANDA_GRIPPER_SPEED * np.sign(action[6]),
                -1.0,
                1.0,
            )

        flat_out[index, :3] = pos
        flat_out[index, 3:6] = ori_ref
        flat_out[index, 6:8] = PANDA_GRIPPER_QPOS_BIAS + PANDA_GRIPPER_QPOS_WEIGHT * gripper_normalized

    return out


def _as_float32(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim < 2:
        raise ValueError(f"{name} must have at least 2 dimensions after batching.")
    return value


def build_full_state_residual_features(
    base_state: np.ndarray,
    actions: np.ndarray,
    proxy_state: np.ndarray,
    delay: int | np.ndarray,
    max_delay: int = 20,
) -> np.ndarray:
    """Build full-state residual features.

    Layout:
    ``[state[q_k](8), right_aligned_actions(max_delay,7), action_mask(max_delay), proxy_state(8), delay/max_delay]``.
    """

    base_state = _as_float32(base_state, "base_state")
    proxy_state = _as_float32(proxy_state, "proxy_state")
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 2:
        actions = actions[None, :, :]
    if base_state.shape[-1] != 8:
        raise ValueError(f"base_state last dimension must be 8, got {base_state.shape[-1]}.")
    if proxy_state.shape[-1] != 8:
        raise ValueError(f"proxy_state last dimension must be 8, got {proxy_state.shape[-1]}.")
    if actions.ndim != 3 or actions.shape[-1] != 7:
        raise ValueError(f"actions must have shape (B, T, 7), got {actions.shape}.")
    if base_state.shape[0] != actions.shape[0] or base_state.shape[0] != proxy_state.shape[0]:
        raise ValueError(
            "Batch dimensions must match: "
            f"base_state={base_state.shape[0]}, actions={actions.shape[0]}, proxy_state={proxy_state.shape[0]}."
        )
    if max_delay <= 0:
        raise ValueError("max_delay must be positive.")

    batch = base_state.shape[0]
    if np.isscalar(delay):
        delay_array = np.full((batch,), int(delay), dtype=np.int64)
    else:
        delay_array = np.asarray(delay, dtype=np.int64).reshape(batch)
    if np.any(delay_array < 0) or np.any(delay_array > max_delay):
        raise ValueError(f"delay values must be in [0, {max_delay}], got {delay_array}.")

    padded_actions = np.zeros((batch, max_delay, actions.shape[-1]), dtype=np.float32)
    action_mask = np.zeros((batch, max_delay), dtype=np.float32)
    for index, delay_value in enumerate(delay_array):
        if delay_value == 0:
            continue
        if actions.shape[1] < delay_value:
            raise ValueError(
                f"actions for sample {index} has length {actions.shape[1]}, "
                f"but delay requires {delay_value} steps."
            )
        padded_actions[index, -delay_value:] = actions[index, -delay_value:]
        action_mask[index, -delay_value:] = 1.0

    delay_feature = delay_array.astype(np.float32).reshape(batch, 1) / float(max_delay)
    return np.concatenate(
        [
            base_state,
            padded_actions.reshape(batch, -1),
            action_mask,
            proxy_state,
            delay_feature,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def build_full_state_residual_targets(proxy_state: np.ndarray, gt_state: np.ndarray) -> np.ndarray:
    """Return ``[delta_pos, relative_delta_rot_axisangle, delta_gripper]`` targets."""

    proxy_state = _as_float32(proxy_state, "proxy_state")
    gt_state = _as_float32(gt_state, "gt_state")
    if proxy_state.shape != gt_state.shape or proxy_state.shape[-1] != 8:
        raise ValueError(f"Expected matching (B, 8) states, got {proxy_state.shape} and {gt_state.shape}.")

    target = np.empty_like(proxy_state, dtype=np.float32)
    target[:, :3] = gt_state[:, :3] - proxy_state[:, :3]
    for index, (proxy, gt) in enumerate(zip(proxy_state, gt_state, strict=True)):
        proxy_rot = axisangle_to_matrix(proxy[3:6])
        gt_rot = axisangle_to_matrix(gt[3:6])
        target[index, 3:6] = matrix_to_axisangle(gt_rot @ proxy_rot.T).astype(np.float32)
    target[:, 6:8] = gt_state[:, 6:8] - proxy_state[:, 6:8]
    return target


def compute_group_residual_scale(target: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Return one scalar std per position/rotation/gripper group, expanded to 8D."""

    target = _as_float32(target, "target")
    if target.shape[-1] != 8:
        raise ValueError(f"target last dimension must be 8, got {target.shape[-1]}.")
    pos_std = max(float(np.std(target[:, :3])), eps)
    rot_std = max(float(np.std(target[:, 3:6])), eps)
    gripper_std = max(float(np.std(target[:, 6:8])), eps)
    return np.asarray([pos_std] * 3 + [rot_std] * 3 + [gripper_std] * 2, dtype=np.float32)


def denormalize_full_state_residual(residual: np.ndarray, residual_scale) -> np.ndarray:
    """Undo the group-scalar-std target normalization.

    Takes the scale array directly (the source took a metadata dict); every call site in this
    bundle already holds the array.
    """
    residual = _as_float32(residual, "residual")
    scale = residual_scale
    if scale is None:
        return residual
    scale_array = np.asarray(scale, dtype=np.float32)
    if scale_array.shape != (8,):
        raise ValueError(f"residual_scale must have shape (8,), got {scale_array.shape}.")
    return residual * scale_array[None, :]


def _canonicalize_axisangle_batch(axisangle: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            canonicalize_axisangle_to_reference(value, ref)
            for value, ref in zip(axisangle, reference, strict=True)
        ],
        axis=0,
    ).astype(np.float32)


def apply_full_state_residual(proxy_state: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Apply full-state residuals using relative rotation composition."""

    proxy_state = _as_float32(proxy_state, "proxy_state")
    residual = _as_float32(residual, "residual")
    if proxy_state.shape[0] != residual.shape[0] or residual.shape[-1] != 8:
        raise ValueError(f"Expected proxy (B, 8) and residual (B, 8), got {proxy_state.shape} and {residual.shape}.")

    corrected = proxy_state.copy()
    corrected[:, :3] = proxy_state[:, :3] + residual[:, :3]
    raw_rot = []
    for proxy, delta in zip(proxy_state, residual, strict=True):
        proxy_rot = axisangle_to_matrix(proxy[3:6])
        delta_rot = axisangle_to_matrix(delta[3:6])
        raw_rot.append(matrix_to_axisangle(delta_rot @ proxy_rot))
    corrected[:, 3:6] = _canonicalize_axisangle_batch(np.asarray(raw_rot, dtype=np.float32), proxy_state[:, 3:6])
    corrected[:, 6:8] = proxy_state[:, 6:8] + residual[:, 6:8]
    return corrected
