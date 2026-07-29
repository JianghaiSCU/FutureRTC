#!/usr/bin/env python3
"""Stage 1 - collect on-policy environment latents for the motion-prior predictor.

Rolls out the per-level BC base policy on-policy. At each replan cycle it records, for a
sampled delay ``d``:
  z_s            - the FULL current observation latent  (predictor input)
  z_target       - the ENVIRONMENT-only latent at the handoff obs s+d  (predictor target)
  motion_actions - the d executed actions, zero-padded to the chunk length
  delay          - d

Writes one ``<level_name>.npz`` per level. The predictor target is the env-contribution
latent (robot dims zeroed), so the predictor learns to fill in the environment part of the
handoff observation while the robot part is recovered by forward-simulation at eval time.

The env predictor MUST be trained on-policy: off-policy / expert data collapses it.

Needs the RTC repo + a BC base policy (see rtc_env / README).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from motion_prior_handoff import rtc_env as rtc  # noqa: E402
from motion_prior_handoff.flow_policy import (  # noqa: E402
    flow_obs_latent,
    level_name_from_path,
    parse_level_indices,
    validate_delay,
)
from motion_prior_handoff.predictor import FORMAT_VERSION  # noqa: E402
from motion_prior_handoff.robot_mask import compute_robot_mask  # noqa: E402


def collect_level_delay(env, env_params, policy, level, robot_mask, *,
                        delay, num_envs, num_cycles, num_flow_steps, seed):
    import jax
    import jax.numpy as jnp
    import kinetix.environment.wrappers as wrappers
    import train_expert

    validate_delay(delay, policy.action_chunk_size)
    benv = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env))), num_envs)

    def scan_actions(rng, obs, env_state, actions):
        def step(carry, action):
            rng, obs, env_state = carry
            rng, key = jax.random.split(rng)
            next_obs, next_env_state, _r, done, _info = benv.step(key, env_state, action, env_params)
            return (rng, next_obs, next_env_state), done
        if actions.shape[1] == 0:
            return (rng, obs, env_state), jnp.zeros((0, obs.shape[0]), dtype=bool)
        return jax.lax.scan(step, (rng, obs, env_state), actions.transpose(1, 0, 2))

    def cycle(carry, _):
        rng, obs, env_state, action_chunk = carry
        z_s = flow_obs_latent(policy, obs)  # full current latent (predictor input)
        step_mask = jnp.arange(policy.action_chunk_size) < delay
        motion_actions = jnp.where(step_mask[None, :, None], action_chunk, 0.0)

        delay_actions = action_chunk[:, :delay]
        (rng, handoff_obs, handoff_env_state), delay_dones = scan_actions(rng, obs, env_state, delay_actions)
        valid = jnp.logical_not(jnp.any(delay_dones, axis=0))
        z_env = flow_obs_latent(policy, jnp.where(robot_mask, 0.0, handoff_obs))     # TARGET (env only)

        rng, key = jax.random.split(rng)
        next_chunk = policy.action(key, handoff_obs, num_flow_steps)
        (rng, next_obs, next_env_state), advance_dones = scan_actions(rng, handoff_obs, handoff_env_state, next_chunk[:, :1])
        shifted = jnp.concatenate(
            [next_chunk[:, 1:], jnp.zeros((num_envs, 1, policy.action_dim), dtype=next_chunk.dtype)], axis=1)
        rng, key = jax.random.split(rng)
        reset_chunk = policy.action(key, next_obs, num_flow_steps)
        crossed = jnp.any(advance_dones, axis=0)
        next_aligned = jnp.where(crossed[:, None, None], reset_chunk, shifted)
        output = {
            "z_s": z_s,
            "z_target": z_env,           # env-contribution latent -> predictor target
            "motion_actions": motion_actions,
            "delay": jnp.full((num_envs,), delay, dtype=jnp.int32),
            "valid": valid,
        }
        return (rng, next_obs, next_env_state, next_aligned), output

    rng = jax.random.key(seed)
    rng, key = jax.random.split(rng)
    obs, env_state = benv.reset_to_level(key, level, env_params)
    rng, key = jax.random.split(rng)
    action_chunk = policy.action(key, obs, num_flow_steps)
    _carry, rows = jax.lax.scan(cycle, (rng, obs, env_state, action_chunk), None, length=num_cycles)
    rows = jax.tree.map(lambda x: np.asarray(x).reshape((-1,) + x.shape[2:]), rows)
    valid = rows.pop("valid").astype(bool)
    return {key: value[valid] for key, value in rows.items()}


def collect(args) -> None:
    import jax

    rtc.bootstrap_rtc(args.rtc_root)
    env, levels, env_params, _ = rtc.load_env_and_levels(max_timesteps=args.max_timesteps, rtc_root=args.rtc_root)
    state_dicts = rtc.load_state_dicts(run_path=args.run_path)
    policy_spec = rtc.build_flow_policy(env, env_params, levels, state_dicts)
    obs_dim = int(policy_spec["obs_dim"])
    delays = [int(v) for v in args.delays.split(",") if v]
    level_indices = parse_level_indices(args.level_indices, len(rtc.DEFAULT_LEVEL_PATHS))
    out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    for li in level_indices:
        level_path = rtc.DEFAULT_LEVEL_PATHS[li]
        level_name = level_name_from_path(level_path)
        level = jax.tree.map(lambda x: x[li], levels)
        policy = rtc._bind(policy_spec, jax.tree.map(lambda x: x[li], state_dicts))
        robot_mask = compute_robot_mask(f"{args.rtc_root}/{level_path}", obs_dim)
        chunks = []
        for delay in delays:
            chunks.append(collect_level_delay(
                env, env_params, policy, level, robot_mask,
                delay=delay, num_envs=args.num_envs, num_cycles=args.num_cycles,
                num_flow_steps=args.num_flow_steps, seed=args.seed + li * 100 + delay))
        arrays = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}
        metadata = {
            "format_version": FORMAT_VERSION, "level_name": level_name, "run_path": args.run_path,
            "base_model": args.base_model, "target_mode": "env_onpolicy",
            "obs_dim": obs_dim, "latent_dim": policy.channel_dim, "action_dim": policy.action_dim,
            "action_chunk_size": policy.action_chunk_size, "delays": delays,
            "action_noise_std": 0.1, "num_valid_samples": int(arrays["z_s"].shape[0]),
        }
        np.savez_compressed(out / f"{level_name}.npz", **arrays, metadata=np.asarray(metadata, dtype=object))
        print(level_name, metadata["num_valid_samples"], flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rtc-root", default=rtc.DEFAULT_RTC_ROOT)
    p.add_argument("--run-path", default=rtc.DEFAULT_RUN_PATH)
    p.add_argument("--base-model", default="bc31", help="tag stored in shard metadata")
    p.add_argument("--level-indices", default="0,1,2,3,4,5,6,7,8,9,10,11")
    p.add_argument("--delays", default="1,2,3,4")
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--num-cycles", type=int, default=64)
    p.add_argument("--num-flow-steps", type=int, default=rtc.NUM_FLOW_STEPS)
    p.add_argument("--max-timesteps", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="outputs/latents")
    collect(p.parse_args())


if __name__ == "__main__":
    main()
