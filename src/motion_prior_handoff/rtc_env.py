"""RTC / Kinetix bootstrap and env + FlowPolicy + checkpoint loaders.

This method builds on the Real-Time Chunking Kinetix codebase
(https://github.com/Physical-Intelligence/real-time-chunking-kinetix). Rather than vendoring
it, we prepend that repo's ``src/`` to ``sys.path`` at runtime and reuse its Kinetix env,
``FlowPolicy`` (``model.py``), and ``train_expert`` helpers. Point ``--rtc-root`` at a checkout
of that repo (with its ``third_party/kinetix`` submodule initialized) and ``--run-path`` at a
per-level BC flow-matching base policy (layout ``<run-path>/<step>/policies/<level>.pkl``;
"bc31" is the epoch-31 checkpoint). Base policies come from ``gs://rtc-assets/bc/`` or are
trained via that repo's ``src/train_flow.py``.
"""
from __future__ import annotations

import os
import pathlib
import pickle
import sys

import jax
import jax.numpy as jnp

# --- Pinned official defaults (do not retune; match RTC's eval_flow.py / model.py) ---
NUM_FLOW_STEPS = 5            # EvalConfig.num_flow_steps
NUM_EVALS_DEFAULT = 2048     # EvalConfig.num_evals
ACTION_CHUNK_SIZE = 8         # ModelConfig.action_chunk_size (= h_max)

DEFAULT_RTC_ROOT = os.environ.get("RTC_ROOT", "RTC")
DEFAULT_RUN_PATH = os.environ.get("RTC_BC_RUN_PATH", str(pathlib.Path(DEFAULT_RTC_ROOT) / "pretrained_bc31"))
DEFAULT_LEVEL_PATHS = (
    "worlds/l/grasp_easy.json", "worlds/l/catapult.json", "worlds/l/cartpole_thrust.json",
    "worlds/l/hard_lunar_lander.json", "worlds/l/mjc_half_cheetah.json", "worlds/l/mjc_swimmer.json",
    "worlds/l/mjc_walker.json", "worlds/l/h17_unicycle.json", "worlds/l/chain_lander.json",
    "worlds/l/catcher_v3.json", "worlds/l/trampoline.json", "worlds/l/car_launch.json",
)


def bootstrap_rtc(rtc_root: str = DEFAULT_RTC_ROOT) -> None:
    """Prepend RTC's src to sys.path so its modules (model, train_expert, kinetix) import."""
    src = str(pathlib.Path(rtc_root) / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def load_env_and_levels(level_paths=DEFAULT_LEVEL_PATHS, max_timesteps=None, rtc_root=DEFAULT_RTC_ROOT):
    """Build the Kinetix env + levels (faithful to RTC's eval setup).

    Level json paths are resolved against ``rtc_root`` so this is cwd-independent.
    """
    import kinetix.environment.env as kenv
    import kinetix.environment.env_state as kenv_state
    import train_expert

    abs_level_paths = [str(pathlib.Path(rtc_root) / p) for p in level_paths]
    static_env_params = kenv_state.StaticEnvParams(
        **train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP
    )
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(abs_level_paths, static_env_params, env_params)
    if max_timesteps is not None:
        env_params = env_params.replace(max_timesteps=max_timesteps)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)
    env = kenv.make_kinetix_env_from_name(
        "Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params
    )
    return env, levels, env_params, static_env_params


def load_state_dicts(run_path=DEFAULT_RUN_PATH, level_paths=DEFAULT_LEVEL_PATHS, step=-1):
    """Load per-level flow-policy params from ``<run-path>/<step>/policies/<level>.pkl``.

    ``step=-1`` selects the highest-numbered checkpoint dir (e.g. ``31`` for bc31).
    """
    state_dicts = []
    for level_path in level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), pathlib.Path(run_path).iterdir()))
        log_dirs = sorted(log_dirs, key=lambda p: int(p.name))
        with (log_dirs[step] / "policies" / f"{level_name}.pkl").open("rb") as f:
            state_dicts.append(pickle.load(f))
    return jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts))


def build_flow_policy(env, env_params, levels, state_dicts):
    """Derive the FlowPolicy construction args (obs_dim, action_dim, model config).

    The concrete policy is built per-level via ``_bind``.
    """
    import model as _model

    obs_dim = jax.eval_shape(
        env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params
    )[0].shape[-1]
    action_dim = env.action_space(env_params).shape[0]
    return {"obs_dim": int(obs_dim), "action_dim": int(action_dim), "model_config": _model.ModelConfig()}


def _bind(policy, state_dict):
    """Build a concrete FlowPolicy from construction args + a level's pure-dict params."""
    import flax.nnx as nnx
    import model as _model

    p = _model.FlowPolicy(
        obs_dim=policy["obs_dim"],
        action_dim=policy["action_dim"],
        config=policy["model_config"],
        rngs=nnx.Rngs(jax.random.key(0)),
    )
    graphdef, state = nnx.split(p)
    state.replace_by_pure_dict(state_dict)
    return nnx.merge(graphdef, state)
