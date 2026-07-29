#!/usr/bin/env python3
"""Stage 3 - evaluate FutureRTC on the Kinetix delay-robustness sweep.

For each (level, delay d, execute_horizon) point: run the policy under an artificial
inference delay. At each replan cycle, when d > 0:
  - forward-simulate the d already-executed actions to get the future observation,
  - take the ROBOT observation latent from that forward-sim future (proprioception is known),
  - take the ENVIRONMENT observation latent from the learned predictor,
  - combine (z_robot + z_env) and decode the next action chunk.
At d = 0 this reduces to the plain no-delay policy.

Reports solve rate and mean execution time (episode length) per delay. Needs the RTC repo,
a BC base policy, and the per-level predictors from Stage 2 (--predictor-dir).
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from motion_prior_handoff import rtc_env as rtc  # noqa: E402
from motion_prior_handoff.delay_grid import (  # noqa: E402
    build_delay_horizon_pairs,
    limit_horizon_pairs_per_delay,
)
from motion_prior_handoff.flow_policy import (  # noqa: E402
    action_from_obs_latent,
    flow_obs_latent,
    level_name_from_path,
    parse_level_indices,
    validate_delay,
)
from motion_prior_handoff.predictor import load_predictor_checkpoint, predict_obs_latent  # noqa: E402
from motion_prior_handoff.results import build_point_record, write_outputs  # noqa: E402
from motion_prior_handoff.robot_mask import compute_robot_mask  # noqa: E402

METHOD = "futurertc"


def eval_handoff(
    env, env_params, policy, level, *,
    robot_mask,
    inference_delay: int,
    execute_horizon: int,
    num_evals: int,
    num_flow_steps: int,
    seed: int,
    predictor_params,
    predictor_metadata,
):
    """One level/delay/horizon point. Robot obs from forward-sim; env obs from the predictor."""
    import jax
    import jax.numpy as jnp
    import kinetix.environment.wrappers as wrappers
    import train_expert

    validate_delay(inference_delay, policy.action_chunk_size)
    if execute_horizon < inference_delay:
        raise ValueError(f"{execute_horizon=} must be >= {inference_delay=}")
    if execute_horizon > policy.action_chunk_size - inference_delay:
        raise ValueError(
            f"{execute_horizon=} must be <= action_chunk_size - inference_delay "
            f"({policy.action_chunk_size - inference_delay})"
        )
    fresh_steps = execute_horizon - inference_delay
    benv = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env))),
        num_evals,
    )

    def scan_actions(rng, obs, env_state, actions):
        def step(carry, action):
            rng, obs, env_state = carry
            rng, key = jax.random.split(rng)
            next_obs, next_env_state, _reward, done, info = benv.step(key, env_state, action, env_params)
            return (rng, next_obs, next_env_state), (done, info)

        return jax.lax.scan(step, (rng, obs, env_state), actions.transpose(1, 0, 2))

    def execute_chunk(carry, _):
        rng, obs, env_state, action_chunk = carry
        rng, key = jax.random.split(rng)
        if inference_delay == 0:
            candidate = policy.action(key, obs, num_flow_steps)
        else:
            # Forward-sim the already-executed delay actions in a SIDE rollout (functional;
            # does not advance the real carry) to get the future obs.
            rng, sim_rng = jax.random.split(rng)
            (_, obs_future, _), _ = scan_actions(
                sim_rng, obs, env_state, action_chunk[:, :inference_delay]
            )
            # robot latent from forward-sim; env latent from predictor; combine (linear).
            z_robot = flow_obs_latent(policy, jnp.where(robot_mask, obs_future, 0.0))
            z_s = flow_obs_latent(policy, obs)
            step_mask = jnp.arange(policy.action_chunk_size) < inference_delay
            motion_actions = jnp.where(step_mask[None, :, None], action_chunk, 0.0)
            delay_tensor = jnp.full((obs.shape[0],), inference_delay, dtype=jnp.int32)
            z_env = predict_obs_latent(
                predictor_params, z_s, motion_actions, delay_tensor,
                max_delay=int(predictor_metadata["max_delay"]),
            )
            candidate = action_from_obs_latent(
                policy, key, z_robot + z_env, num_flow_steps, int(predictor_metadata["obs_dim"])
            )
        actions = jnp.concatenate(
            [action_chunk[:, :inference_delay], candidate[:, :fresh_steps]], axis=1
        )
        (rng, next_obs, next_env_state), (dones, infos) = scan_actions(rng, obs, env_state, actions)
        shifted = jnp.concatenate(
            [
                candidate[:, fresh_steps:],
                jnp.zeros((obs.shape[0], fresh_steps, policy.action_dim), dtype=candidate.dtype),
            ],
            axis=1,
        )
        return (rng, next_obs, next_env_state, shifted), (dones, infos)

    rng = jax.random.key(seed)
    rng, key = jax.random.split(rng)
    obs, env_state = benv.reset_to_level(key, level, env_params)
    rng, key = jax.random.split(rng)
    action_chunk = policy.action(key, obs, num_flow_steps)
    scan_length = math.ceil(env_params.max_timesteps / execute_horizon)
    _, (dones, infos) = jax.lax.scan(
        execute_chunk, (rng, obs, env_state, action_chunk), None, length=scan_length
    )
    dones, infos = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), (dones, infos))
    first_done_idx = jnp.argmax(dones, axis=0)
    idx = (first_done_idx, jnp.arange(num_evals))
    return infos["returned_episode_solved"][idx].mean(), infos["returned_episode_lengths"][idx].mean()


def run(args):
    import jax

    rtc.bootstrap_rtc(args.rtc_root)
    env, levels, env_params, _static = rtc.load_env_and_levels(
        max_timesteps=args.max_timesteps, rtc_root=args.rtc_root
    )
    state_dicts = rtc.load_state_dicts(run_path=args.run_path)
    policy_spec = rtc.build_flow_policy(env, env_params, levels, state_dicts)
    obs_dim = int(policy_spec["obs_dim"])
    delays = [int(v) for v in args.delays.split(",") if v]
    pairs = build_delay_horizon_pairs(rtc.ACTION_CHUNK_SIZE, delays)
    pairs = limit_horizon_pairs_per_delay(pairs, args.max_horizons_per_delay)
    level_indices = parse_level_indices(args.level_indices, len(rtc.DEFAULT_LEVEL_PATHS))

    level_items = []
    for li in level_indices:
        level_path = rtc.DEFAULT_LEVEL_PATHS[li]
        level_name = level_name_from_path(level_path)
        level = jax.tree.map(lambda x: x[li], levels)
        policy = rtc._bind(policy_spec, jax.tree.map(lambda x: x[li], state_dicts))
        robot_mask = compute_robot_mask(f"{args.rtc_root}/{level_path}", obs_dim)
        predictor_params, predictor_metadata = load_predictor_checkpoint(
            pathlib.Path(args.predictor_dir) / f"{level_name}.pkl"
        )
        level_items.append((level_name, level, policy, robot_mask, predictor_params, predictor_metadata))

    records = []
    for delay, execute_horizon in pairs:
        for level_name, level, policy, robot_mask, predictor_params, predictor_metadata in level_items:
            solve_rate, mean_length = eval_handoff(
                env, env_params, policy, level,
                robot_mask=robot_mask,
                inference_delay=delay, execute_horizon=execute_horizon,
                num_evals=args.num_evals, num_flow_steps=args.num_flow_steps, seed=args.seed,
                predictor_params=predictor_params, predictor_metadata=predictor_metadata,
            )
            records.append(build_point_record(
                method=METHOD, delay=delay, execute_horizon=execute_horizon,
                level_name=level_name, solve_rate=float(solve_rate),
                n_trials=args.num_evals, execution_time=float(mean_length), seed=args.seed,
                base_model=args.base_model,
            ))
            print(METHOD, level_name, delay, execute_horizon, float(solve_rate), float(mean_length), flush=True)
        write_outputs(args.output_dir, records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtc-root", default=rtc.DEFAULT_RTC_ROOT)
    parser.add_argument("--run-path", default=rtc.DEFAULT_RUN_PATH)
    parser.add_argument("--base-model", default="bc31", help="tag stored in the output records")
    parser.add_argument("--level-indices", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--predictor-dir", default="weights/predictors",
                        help="per-level predictor checkpoints (defaults to the shipped weights)")
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--delays", default="0,1,2,3,4")
    parser.add_argument("--num-evals", type=int, default=rtc.NUM_EVALS_DEFAULT)
    parser.add_argument("--max-horizons-per-delay", type=int)
    parser.add_argument("--max-timesteps", type=int)
    parser.add_argument("--num-flow-steps", type=int, default=rtc.NUM_FLOW_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
