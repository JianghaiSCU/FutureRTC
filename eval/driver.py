#!/usr/bin/env python3
"""Multi-delay LIBERO evaluation of the predictor + corrector method.

At each handoff the policy needs an observation it cannot have: the delay ``d`` means the freshest
image available is ``d`` steps old. The method supplies an estimate of the missing observation --
the predictor forecasts the visual latent forward by ``d`` steps, the corrector estimates the
proprioceptive state at the handoff -- so the policy is queried with an approximation of the
handoff observation and executes its fresh chunk from index 0.

All requested delays run CONCURRENTLY in one async vector-env batch (``n_envs = b_t * n_delays``),
so a whole delay sweep costs one pass of env stepping on one GPU rather than one pass per delay.

Two action spaces meet in ``_query`` and must never be confused: the corrector integrates PHYSICAL
OSC deltas (ENV), while the predictor's motion prior was trained on the latent bank's
quantile-normalized actions (BANK). They are kept as two explicitly named locals.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import pathlib
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.action_space import env_to_qnorm, load_action_stats  # noqa: E402
from common.latent_io import capture_latent, inject_latent  # noqa: E402
from common.runtime import build_runtime, normalize_single_visible_egl_device  # noqa: E402
from corrector.state_fn import Corrector  # noqa: E402
from eval.async_env import make_eval_envs  # noqa: E402
from eval.metrics import EpisodeRecord, write_outputs  # noqa: E402
from eval.schedule import (  # noqa: E402
    committed_slice, compute_delay_batch_layout, stale_step, step_index,
)
from predictor.model import forecast, load_predictor  # noqa: E402

METHOD = "predictor_corrector"


# --------------------------------------------------------------------------------------
# env / batch helpers
# --------------------------------------------------------------------------------------

def slice_raw_obs(raw_observation: dict, lo: int, hi: int) -> dict:
    """Slice a raw vector-env observation along the env axis, recursing into nested dicts."""
    def _slice(v):
        if isinstance(v, (str, bytes)):
            return v
        if isinstance(v, dict):
            return {kk: _slice(vv) for kk, vv in v.items()}
        if getattr(v, "shape", None) is not None:
            return v[lo:hi]
        if isinstance(v, (list, tuple)):
            return type(v)(v[lo:hi])
        return v
    return {k: _slice(v) for k, v in raw_observation.items()}


def assemble_action(t, s, delays, chunks_by_group, b_t) -> np.ndarray:
    """One action column per delay group -> [N, 1, A]. Every group executes its chunk from 0."""
    k = t // s
    cols = []
    for g in range(len(delays)):
        chunk = chunks_by_group[g][k]
        idx = min(step_index(t, s), chunk.shape[1] - 1)
        cols.append(chunk[:, idx, :])
    return np.concatenate(cols, axis=0)[:, None, :]


def _success_array(info: dict, batch_size: int) -> np.ndarray:
    value = info.get("is_success")
    if value is None and "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict):
            value = final_info.get("is_success")
        else:
            value = [bool(item.get("is_success", False)) if item else False for item in final_info]
    if value is None:
        return np.zeros(batch_size, dtype=bool)
    arr = np.asarray(value, dtype=bool)
    if arr.shape == ():
        return np.full(batch_size, bool(arr), dtype=bool)
    out = np.zeros(batch_size, dtype=bool)
    out[: min(batch_size, arr.size)] = arr.reshape(-1)[:batch_size]
    return out


def _set_vector_init_state_ids(env, init_state_ids: list[int]) -> None:
    if hasattr(env, "set_attr"):
        env.set_attr("init_state_id", [int(x) for x in init_state_ids])
        return
    envs_list = getattr(env, "envs", None) or getattr(getattr(env, "unwrapped", None), "envs", None)
    if not envs_list:
        raise RuntimeError("Vector env does not support init_state_id assignment")
    for sub_env, init_state_id in zip(envs_list, init_state_ids):
        sub_env.init_state_id = int(init_state_id)


def _step_batch_actions(env, actions, active, success, done, step_counts, max_steps):
    """Step a vector env with an action sequence [B, T, A], updating per-env episode state."""
    if actions.shape[1] == 0:
        return None, success, done, step_counts
    obs = None
    batch_size = actions.shape[0]
    zero_action = np.zeros(actions.shape[-1], dtype=actions.dtype)
    for t in range(actions.shape[1]):
        active = active & ~done & (step_counts < max_steps)
        if not np.any(active):
            break
        step_actions = actions[:, t, :].copy()
        step_actions[~active] = zero_action
        obs, _, terminated, truncated, info = env.step(step_actions)
        step_counts[active] += 1
        terminated = np.asarray(terminated, dtype=bool).reshape(-1)[:batch_size]
        truncated = np.asarray(truncated, dtype=bool).reshape(-1)[:batch_size]
        newly_done = active & (terminated | truncated | (step_counts >= max_steps))
        success = success | (newly_done & _success_array(info, batch_size))
        done = done | newly_done
    return obs, success, done, step_counts


def _add_envs_task(runtime, env, observation):
    """Add task strings for both Sync and Async vector envs.

    LeRobot 0.5.1's add_envs_task reads ``env.envs[0]``, which an AsyncVectorEnv does not expose;
    in that case ask the workers via ``call()``.
    """
    if hasattr(env, "envs"):
        return runtime.raw["add_envs_task"](env, observation)
    task_result = None
    last_error = None
    for attr in ("task_description", "task"):
        try:
            task_result = env.call(attr)
            break
        except Exception as exc:  # noqa: BLE001 - try the fallback attribute
            last_error = exc
    if task_result is None:
        if last_error is not None:
            raise last_error
        num_envs = getattr(env, "num_envs", None)
        if num_envs is None:
            first = observation[next(iter(observation))]
            num_envs = int(getattr(first, "shape", [1])[0])
        observation["task"] = ["" for _ in range(int(num_envs))]
        return observation
    if isinstance(task_result, tuple):
        task_result = list(task_result)
    if not isinstance(task_result, list) or not all(isinstance(x, str) for x in task_result):
        raise TypeError(f"expected a list of task strings, got {type(task_result)}")
    observation["task"] = task_result
    return observation


def _prepare_group_batch(runtime, raw_obs_slice, task_list, state_override=None):
    observation = runtime.raw["preprocess_observation"](raw_obs_slice)
    observation["task"] = list(task_list)
    observation = runtime.env_preprocessor(observation)
    if state_override is not None:
        observation[runtime.obs_state_key] = state_override
    return runtime.preprocessor(observation), observation


def _predict_chunk(runtime, batch):
    """One policy chunk prediction -> (env-space action chunk [B, H, A], inference ms)."""
    torch = runtime.raw["torch"]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        raw = runtime.policy.predict_action_chunk(batch)
        env_actions = runtime.postprocessor(raw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return env_actions.detach().cpu().numpy(), (time.perf_counter() - start) * 1000.0


# --------------------------------------------------------------------------------------
# episode loop
# --------------------------------------------------------------------------------------

def run_episode_batch(runtime, *, env, delays, preinfer_steps, max_steps, seed, b_t,
                      init_state_ids, predictor, latent_norm, corrector, action_stats):
    """Run one block of episodes with every delay stepping concurrently."""
    torch = runtime.raw["torch"]
    backbone = runtime.backbone
    s = int(preinfer_steps)
    G = len(delays)
    N = b_t * G

    runtime.policy.reset()
    block_seed = (int(seed) * 1000003 + int(init_state_ids[0])) % 2147483647
    torch.manual_seed(block_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(block_seed)
    _set_vector_init_state_ids(env, list(init_state_ids) * G)
    raw_observation, _ = env.reset(seed=[seed] * N)
    full_tasks = list(_add_envs_task(
        runtime, env, runtime.raw["preprocess_observation"](raw_observation))["task"])

    step_counts = np.zeros(N, dtype=int)
    success = np.zeros(N, dtype=bool)
    done = np.zeros(N, dtype=bool)
    active = np.ones(N, dtype=bool)

    chunks_by_group = {g: {} for g in range(G)}
    obs_buf = {g: {} for g in range(G)}      # env time -> raw obs slice (the stale image source)
    z_init_cache = {}                        # g -> the episode's first-frame latent

    def _group_slice(g):
        return g * b_t, (g + 1) * b_t

    def _episode_z_init(g, device):
        if g not in z_init_cache:
            lo, hi = _group_slice(g)
            batch0, _ = _prepare_group_batch(runtime, obs_buf[g][0], full_tasks[lo:hi])
            z_init_cache[g] = capture_latent(backbone, runtime.policy, batch0).to(device)
        return z_init_cache[g]

    def _query(g, window_m):
        d = delays[g]
        lo, hi = _group_slice(g)
        group_tasks = full_tasks[lo:hi]
        current_raw = slice_raw_obs(raw_observation, lo, hi)
        stale_t = stale_step(window_m, s, d)
        have_context = (window_m >= 1 and d > 0 and stale_t in obs_buf[g]
                        and (window_m - 1) in chunks_by_group[g])

        if not have_context:
            # d = 0, window 0, or missing context: the fresh observation IS the handoff
            # observation, so no prediction is needed. This is the no-delay baseline.
            batch, _ = _prepare_group_batch(runtime, current_raw, group_tasks)
            chunks_by_group[g][window_m] = np.asarray(_predict_chunk(runtime, batch)[0])
            return

        lo_c, hi_c = committed_slice(s, d)
        # ENV: physical OSC deltas straight from the policy postprocessor -- what the corrector
        # forward-integrates.
        committed_env = chunks_by_group[g][window_m - 1][:, lo_c:hi_c, :]
        # BANK: quantile-normalized -- the space the predictor's motion prior was trained on.
        committed_bank = np.asarray(env_to_qnorm(committed_env, action_stats), dtype=np.float32)

        _, stale_proc = _prepare_group_batch(runtime, obs_buf[g][stale_t], group_tasks)
        base_state8 = stale_proc[runtime.obs_state_key].reshape(hi - lo, -1)[:, :8].float()
        corrected = corrector.correct_batch(base_state8.cpu().numpy(), committed_env, d)
        state_override = stale_proc[runtime.obs_state_key].clone()
        state_override.reshape(hi - lo, -1)[:, :8] = torch.as_tensor(
            corrected, dtype=state_override.dtype, device=state_override.device)

        batch, _ = _prepare_group_batch(runtime, obs_buf[g][stale_t], group_tasks,
                                        state_override=state_override)
        z_stale = capture_latent(backbone, runtime.policy, batch)
        z_hat = forecast(predictor, latent_norm, z_stale, committed_bank, d,
                         z_init=_episode_z_init(g, z_stale.device),
                         state=state_override.reshape(hi - lo, -1)[:, :8].float())
        with inject_latent(backbone, runtime.policy, z_hat):
            chunks_by_group[g][window_m] = np.asarray(_predict_chunk(runtime, batch)[0])

    for g in range(G):
        obs_buf[g][0] = slice_raw_obs(raw_observation, *_group_slice(g))
    for g in range(G):
        _query(g, 0)

    t = 0
    while np.any(~done & (step_counts < max_steps)):
        k = t // s
        for g in range(G):
            if k not in chunks_by_group[g]:
                _query(g, k)
        action = assemble_action(t, s, delays, chunks_by_group, b_t)
        raw_observation, success, done, step_counts = _step_batch_actions(
            env, action, active, success, done, step_counts, max_steps)
        t += 1
        for g in range(G):
            obs_buf[g][t] = slice_raw_obs(raw_observation, *_group_slice(g))
            for old in [j for j in obs_buf[g] if 0 < j < t - delays[g] - 1]:
                del obs_buf[g][old]      # only the last d+1 observations can still be needed
        # the chunk for window m is queried at t = m*s, i.e. right after stepping onto the handoff
        if t % s == 0 and np.any(~done & (step_counts < max_steps)):
            for g in range(G):
                if (t // s) not in chunks_by_group[g]:
                    _query(g, t // s)

    return success.tolist(), step_counts.tolist()


# --------------------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------------------

def _record_from_row(row: dict) -> EpisodeRecord:
    return EpisodeRecord(
        benchmark=row["benchmark"], base_model=row["base_model"], method=row["method"],
        delay=row["delay"], task_or_level_id=row["task_or_level_id"],
        episode_idx=row["episode_idx"], solve_rate=row["solve_rate"], n_trials=row["n_trials"],
        execution_time=row["execution_time"], seed=row["seed"],
        execute_horizon=row["execute_horizon"])


def run(args):
    normalize_single_visible_egl_device()
    runtime = build_runtime(
        args.backbone, policy_path=args.policy_path, lerobot_root=args.lerobot_root,
        device=args.device, tokenizer_path=args.tokenizer_path,
        task_suite_name=args.task_suite_name, task_ids=args.task_ids, max_steps=args.max_steps,
        control_mode=args.control_mode)
    predictor, latent_norm = load_predictor(args.predictor_checkpoint, device=args.device)
    corrector = Corrector(args.corrector_checkpoint)
    action_stats = load_action_stats(args.action_stats)

    suite = args.task_suite_name
    delays = [int(x) for x in args.delays.split(",")]
    task_ids = [int(x) for x in args.task_ids.split(",")]
    base_model = runtime.backbone

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, done_keys = [], set()
    results_path = output_dir / "results.jsonl"
    if args.resume and results_path.exists():
        for line in results_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(_record_from_row(row))
            done_keys.add((row["benchmark"], row["base_model"], row["method"], row["delay"],
                           row["task_or_level_id"], row["episode_idx"]))
        print(f"[resume] {len(records)} episodes already done", flush=True)

    nproc = multiprocessing.cpu_count()
    b_t = compute_delay_batch_layout(len(delays), args.num_trials_per_task, nproc,
                                     args.cpu_fraction)
    n_envs = b_t * len(delays)
    print(f"[eval] {METHOD} backbone={base_model} suite={suite} delays={delays} "
          f"b_t={b_t} n_envs={n_envs} (cpu_fraction={args.cpu_fraction}, nproc={nproc})",
          flush=True)

    for task_id in task_ids:
        runtime.env_cfg.task_ids = [task_id]
        envs = make_eval_envs(runtime.raw, runtime.env_cfg, lerobot_root=args.lerobot_root,
                              n_envs=n_envs, use_async_envs=True)
        env = envs[suite][task_id]
        try:
            for block_start in range(0, args.num_trials_per_task, b_t):
                block = list(range(block_start,
                                   min(block_start + b_t, args.num_trials_per_task)))
                padded = block + [block[-1]] * (b_t - len(block))
                pending = {(d, e) for d in delays for e in block
                           if (suite, base_model, METHOD, d, str(task_id), e) not in done_keys}
                if not pending:
                    continue
                succ, steps = run_episode_batch(
                    runtime, env=env, delays=delays, preinfer_steps=args.preinfer_steps,
                    max_steps=args.max_steps, seed=args.seed, b_t=b_t, init_state_ids=padded,
                    predictor=predictor, latent_norm=latent_norm, corrector=corrector,
                    action_stats=action_stats)
                for g, d in enumerate(delays):
                    for j, e in enumerate(block):
                        if (d, e) not in pending:
                            continue
                        local = g * b_t + j
                        records.append(EpisodeRecord(
                            benchmark=suite, base_model=base_model, method=METHOD, delay=d,
                            task_or_level_id=str(task_id), episode_idx=e,
                            solve_rate=1.0 if succ[local] else 0.0, n_trials=1,
                            execution_time=float(steps[local]), seed=args.seed,
                            execute_horizon=None))
                        done_keys.add((suite, base_model, METHOD, d, str(task_id), e))
                write_outputs(output_dir, records)
        finally:
            runtime.raw["close_envs"](envs)
    return records


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", required=True, choices=["pi05", "smolvla"])
    p.add_argument("--policy-path", required=True)
    p.add_argument("--predictor-checkpoint", required=True)
    p.add_argument("--corrector-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--delays", default="5,10,15,20")
    p.add_argument("--task-suite-name", default="libero_spatial")
    p.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--num-trials-per-task", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=220)
    p.add_argument("--preinfer-steps", type=int, default=25, help="replan stride s")
    p.add_argument("--lerobot-root", default=None,
                   help="LeRobot source checkout (only needed when LeRobot is not "
                        "installed in this environment); also read from LEROBOT_ROOT")
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--action-stats", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--control-mode", default="relative")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cpu-fraction", type=float, default=0.32,
                   help="share of CPU cores to spend on env workers; n_envs = b_t * n_delays")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
