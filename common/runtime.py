"""Backbone policy + preprocessor construction, and the single-GPU EGL fix.

Shared by the latent collector, the policy-distillation loss, and the eval driver, so all three
build the policy exactly the same way and therefore see identical visual latents.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import pathlib
import sys
import types

from common.backbones import get_backbone, register_all_configs


def normalize_single_visible_egl_device() -> None:
    """Reconcile robosuite's import-time EGL device check with EGL's post-remap device id.

    robosuite asserts at import that ``MUJOCO_EGL_DEVICE_ID`` is one of the PHYSICAL ids in
    ``CUDA_VISIBLE_DEVICES``, while EGL later expects the remapped in-process id ``0``. The caller
    (a launch script) sets ``MUJOCO_EGL_DEVICE_ID`` to the physical id; this imports robosuite so
    the assert sees a match, then rewrites the variable to ``0``. Do not reorder those two steps.

    No-op unless exactly one GPU is visible AND the caller already set the variable to match it.
    """
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    egl_device = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    if "," in visible_devices or not visible_devices or egl_device != visible_devices:
        return

    import robosuite.utils.binding_utils  # noqa: F401  (triggers the import-time assert)

    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    print(f"[runtime] normalized MUJOCO_EGL_DEVICE_ID {egl_device}->0 "
          f"for single visible GPU {visible_devices}", flush=True)


def resolve_lerobot_root(lerobot_root: str | pathlib.Path | None = None) -> pathlib.Path | None:
    """Where to find LeRobot, or None when it is already importable.

    Resolution order:

    1. an explicit ``--lerobot-root`` / argument,
    2. the ``LEROBOT_ROOT`` environment variable,
    3. nothing -- LeRobot is expected to be installed in the environment (pip, uv, conda).

    A source checkout is only needed when LeRobot is not installed; this keeps the bundle free of
    any machine-specific path.
    """
    candidate = lerobot_root or os.environ.get("LEROBOT_ROOT") or None
    return pathlib.Path(candidate).expanduser().resolve() if candidate else None


def _append_lerobot_src(lerobot_root: str | pathlib.Path | None) -> None:
    """Put a LeRobot source checkout on sys.path, unless LeRobot already imports."""
    root = resolve_lerobot_root(lerobot_root)
    if root is None:
        if importlib.util.find_spec("lerobot") is not None:
            return  # installed in the environment; nothing to do
        raise FileNotFoundError(
            "LeRobot is not importable and no source checkout was given. Install LeRobot in this "
            "environment, or pass --lerobot-root / set LEROBOT_ROOT to a checkout containing src/.")
    src = root / "src"
    if not src.exists():
        raise FileNotFoundError(f"LeRobot src directory not found: {src}")
    src_s = str(src)
    if src_s not in sys.path:
        sys.path.insert(0, src_s)


def _lazy_import_lerobot(lerobot_root: str | pathlib.Path | None = None) -> dict:
    """Import the LeRobot symbols the bundle needs, after making LeRobot importable."""
    _append_lerobot_src(lerobot_root)

    import gymnasium as gym
    import torch

    register_all_configs()  # registers pi05/smolvla configs with draccus before any from_pretrained
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.constants import OBS_STATE

    return {
        "gym": gym,
        "torch": torch,
        "PreTrainedConfig": PreTrainedConfig,
        "LiberoEnvConfig": LiberoEnvConfig,
        "make_env": make_env,
        "make_env_pre_post_processors": make_env_pre_post_processors,
        "add_envs_task": add_envs_task,
        "close_envs": close_envs,
        "preprocess_observation": preprocess_observation,
        "make_policy": make_policy,
        "make_pre_post_processors": make_pre_post_processors,
        "OBS_STATE": OBS_STATE,
    }


def _parse_int_list(value) -> list[int]:
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(v) for v in value]


def _build_from_lerobot(raw: dict, args):
    """Construct the LIBERO env config, the policy, and the three processors."""
    torch = raw["torch"]
    policy_cfg = raw["PreTrainedConfig"].from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = args.device
    policy_cfg.use_amp = args.use_amp

    env_cfg = raw["LiberoEnvConfig"](
        task=args.task_suite_name,
        task_ids=_parse_int_list(args.task_ids),
        obs_type="pixels_agent_pos",
        control_mode=args.control_mode,
        episode_length=args.max_steps,
    )

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # No RTC config is created here, deliberately. RTC is not a method of this bundle, and the
    # policy's `_use_rtc()` is `rtc_config is not None and rtc_config.enabled` -- so leaving it
    # None is equivalent to building an RTCConfig and disabling it, without the dead object.
    # Do not "restore" it: an RTCConfig whose `enabled` is ever flipped on would silently change
    # what the eval measures.

    # A non-empty rename_map must reach make_policy so it (a) skips
    # validate_visual_features_consistency, which compares the env's raw feature names against the
    # policy's expected camera1/2/3 and otherwise hard-fails on the 3-camera video SmolVLA policy,
    # and (b) wires the rename_observations_processor.
    rename_map = args.rename_map or {}
    policy = raw["make_policy"](cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map or None)
    policy.eval()

    env_preprocessor, _ = raw["make_env_pre_post_processors"](
        env_cfg=env_cfg, policy_cfg=policy_cfg)
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": rename_map},
    }
    # pi0.5 uses the external PaliGemma tokenizer; SmolVLA ships its own SmolVLM tokenizer, so its
    # tokenizer_override is None and the checkpoint's preprocessor is kept untouched.
    if args.tokenizer_override is not None:
        preprocessor_overrides["tokenizer_processor"] = {
            "tokenizer_name": args.tokenizer_override}
    preprocessor, postprocessor = raw["make_pre_post_processors"](
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    return env_cfg, policy, env_preprocessor, preprocessor, postprocessor


@dataclasses.dataclass
class Runtime:
    raw: dict
    env_cfg: object
    policy: object
    env_preprocessor: object
    preprocessor: object
    postprocessor: object
    obs_state_key: str
    backbone: str


def build_runtime(
    backbone: str,
    *,
    policy_path: str,
    lerobot_root: str | None = None,
    device: str = "cuda",
    tokenizer_path: str | None = None,
    task_suite_name: str = "libero_spatial",
    task_ids: str = "0",
    max_steps: int = 220,
    control_mode: str = "relative",
) -> Runtime:
    """Load the backbone policy and its LIBERO preprocessors.

    ``task_suite_name`` / ``task_ids`` / ``max_steps`` only shape the env config LeRobot builds
    alongside the policy; callers that never step an env (the collector, the policy loss) can
    leave them at their defaults.
    """
    spec = get_backbone(backbone)
    raw = _lazy_import_lerobot(lerobot_root)

    print(f"[runtime] loading {spec.name} policy from {policy_path}", flush=True)
    args = types.SimpleNamespace(
        policy_path=policy_path,
        device=device,
        control_mode=control_mode,
        rename_map=spec.rename_map,
        tokenizer_override=tokenizer_path if spec.external_tokenizer else None,
        use_amp=False,
        task_suite_name=task_suite_name,
        task_ids=task_ids,
        max_steps=max_steps,
    )
    env_cfg, policy, env_pre, pre, post = _build_from_lerobot(raw, args)
    print("[runtime] ready", flush=True)
    return Runtime(
        raw=raw, env_cfg=env_cfg, policy=policy, env_preprocessor=env_pre,
        preprocessor=pre, postprocessor=post, obs_state_key=raw["OBS_STATE"], backbone=spec.name,
    )
