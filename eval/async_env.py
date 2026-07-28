"""Spawn-safe LIBERO vector-env construction.

LeRobot's LIBERO factory builds AsyncVectorEnv with the platform default ``fork`` context. Here the
CUDA policy is already constructed before env creation, so forking can deadlock; the env callables
below are top-level and pickleable, which allows ``context="spawn"`` instead.
"""
from __future__ import annotations

import pathlib
import sys
from functools import partial
from typing import Any

ASYNC_ENV_CONTEXT = "spawn"


def _append_lerobot_src(lerobot_root: str | pathlib.Path | None) -> None:
    """Put a LeRobot source checkout on sys.path, unless LeRobot already imports.

    ``lerobot_root`` is None whenever LeRobot is installed in the environment. The logic is
    duplicated from ``common.runtime`` rather than imported, because this module also runs inside
    spawn workers, where the fewest possible imports before robosuite is the safer choice.
    """
    import importlib.util
    import os

    root = lerobot_root or os.environ.get("LEROBOT_ROOT") or None
    if root is None:
        if importlib.util.find_spec("lerobot") is not None:
            return
        raise FileNotFoundError(
            "LeRobot is not importable and no source checkout was given; "
            "install LeRobot or set LEROBOT_ROOT / pass --lerobot-root.")
    src = pathlib.Path(root) / "src"
    if not src.exists():
        raise FileNotFoundError(f"LeRobot src directory not found: {src}")
    src_s = str(src)
    if src_s not in sys.path:
        sys.path.insert(0, src_s)


def _normalize_worker_egl_device() -> None:
    """Make robosuite's import-time EGL assert pass inside a spawn worker.

    Spawn workers inherit ``MUJOCO_EGL_DEVICE_ID=0`` (the parent rewrote it to the
    in-process remapped id after its own robosuite import) together with a possibly
    non-zero ``CUDA_VISIBLE_DEVICES``. robosuite asserts at import that
    ``MUJOCO_EGL_DEVICE_ID`` is one of the physical ids in ``CUDA_VISIBLE_DEVICES``,
    so the inherited ``0`` fails on any GPU != 0. Restore the physical id, import
    robosuite so the assert sees a match, then rewrite to the remapped id ``0`` that
    EGL expects in-process. No-op for multi-GPU / unset CUDA_VISIBLE_DEVICES.
    """
    import os

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        return
    os.environ["MUJOCO_EGL_DEVICE_ID"] = visible
    import robosuite.utils.binding_utils  # noqa: F401  (triggers the import-time assert)

    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"


def make_libero_env(
    *,
    lerobot_root: str | pathlib.Path,
    suite_name: str,
    task_id: int,
    episode_index: int,
    n_envs: int,
    camera_name: str,
    init_states: bool,
    episode_length: int | None,
    control_mode: str,
    gym_kwargs: dict[str, Any],
):
    # Runs in the spawn worker: fix the EGL device id BEFORE importing
    # lerobot.envs.libero (which imports robosuite) so multi-GPU async works.
    _normalize_worker_egl_device()
    _append_lerobot_src(lerobot_root)
    from lerobot.envs import libero as libero_envs

    local_kwargs = dict(gym_kwargs)
    local_kwargs.pop("task_ids", None)
    suite = libero_envs._get_suite(suite_name)
    return libero_envs.LiberoEnv(
        task_suite=suite,
        task_id=int(task_id),
        task_suite_name=suite_name,
        camera_name=camera_name,
        init_states=bool(init_states),
        episode_length=episode_length,
        episode_index=int(episode_index),
        n_envs=int(n_envs),
        control_mode=control_mode,
        **local_kwargs,
    )


def make_spawn_safe_libero_env_fns(
    *,
    lerobot_root: str | pathlib.Path,
    suite_name: str,
    task_id: int,
    n_envs: int,
    camera_name: str,
    init_states: bool,
    episode_length: int | None,
    control_mode: str,
    gym_kwargs: dict[str, Any],
):
    clean_gym_kwargs = dict(gym_kwargs)
    clean_gym_kwargs.pop("task_ids", None)
    return [
        partial(
            make_libero_env,
            lerobot_root=None if lerobot_root is None else str(lerobot_root),
            suite_name=str(suite_name),
            task_id=int(task_id),
            episode_index=episode_index,
            n_envs=int(n_envs),
            camera_name=str(camera_name),
            init_states=bool(init_states),
            episode_length=episode_length,
            control_mode=str(control_mode),
            gym_kwargs=clean_gym_kwargs,
        )
        for episode_index in range(int(n_envs))
    ]


def make_spawn_safe_libero_envs(
    runtime: dict[str, Any],
    env_cfg: Any,
    *,
    lerobot_root: str | pathlib.Path,
    n_envs: int,
):
    _append_lerobot_src(lerobot_root)
    from lerobot.envs.libero import _get_suite, _select_task_ids

    gym = runtime["gym"]
    gym_kwargs = dict(env_cfg.gym_kwargs)
    task_ids_filter = gym_kwargs.get("task_ids")
    suite_names = [s.strip() for s in str(env_cfg.task).split(",") if s.strip()]
    if not suite_names:
        raise ValueError("`task` must contain at least one LIBERO suite name.")

    print(
        "Creating LIBERO async envs | "
        f"suites={suite_names} | n_envs(per task)={n_envs} | "
        f"init_states={env_cfg.init_states} | context={ASYNC_ENV_CONTEXT}",
        flush=True,
    )
    if task_ids_filter is not None:
        print(f"Restricting to task_ids={task_ids_filter}", flush=True)

    out: dict[str, dict[int, Any]] = {}
    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        selected = _select_task_ids(len(suite.tasks), task_ids_filter)
        out[suite_name] = {}
        for task_id in selected:
            fns = make_spawn_safe_libero_env_fns(
                lerobot_root=lerobot_root,
                suite_name=suite_name,
                task_id=task_id,
                n_envs=n_envs,
                camera_name=env_cfg.camera_name,
                init_states=env_cfg.init_states,
                episode_length=env_cfg.episode_length,
                control_mode=env_cfg.control_mode,
                gym_kwargs=gym_kwargs,
            )
            out[suite_name][task_id] = gym.vector.AsyncVectorEnv(
                fns,
                context=ASYNC_ENV_CONTEXT,
            )
            print(
                f"Built async vec env | suite={suite_name} | task_id={task_id} | n_envs={n_envs}",
                flush=True,
            )
    return out


def make_eval_envs(runtime, env_cfg, *, lerobot_root, n_envs: int, use_async_envs: bool = True):
    """Build the per-(suite, task) vector envs the driver steps."""
    if use_async_envs:
        return make_spawn_safe_libero_envs(
            runtime, env_cfg, lerobot_root=lerobot_root, n_envs=n_envs)
    return runtime["make_env"](env_cfg, n_envs=n_envs, use_async_envs=False)
