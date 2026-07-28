#!/usr/bin/env python3
"""Build the per-frame visual-latent bank for either backbone.

Each selected LIBERO demonstration frame is embedded exactly ONCE, at the same latent boundary the
eval driver injects at (``common.latent_io.capture_latent``), so training and deployment see
identical latents by construction. Delays are resampled at training time from this bank, which
gives full (frame x delay) coverage with no per-pair duplication.

Actions are stored in BANK space -- ``qnorm(env) = 2*(a-q01)/(q99-q01) - 1``. This is the single
point where that convention enters the pipeline, and it goes through ``common.action_space`` so it
cannot drift from the corrector's inverse.

Storage dtype defaults to bfloat16: the deployed latent is bf16 (the prefix concatenation forces
it), so a bf16 bank is both faithful and train/eval-consistent -- fp32 storage doubles the disk for
precision the model discards.

Large banks can be collected in parallel: run one process per ``--worker-id`` on its own GPU with
the same ``--output-dir``. Episodes are partitioned by ``episode_index % num_workers``, and
``--shard-index-stride`` keeps each worker's shard names unique AND lexicographically ordered,
which is the bank's global frame order.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time as _time

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.action_space import env_to_qnorm, load_action_stats  # noqa: E402
from common.backbones import get_backbone  # noqa: E402
from common.latent_io import capture_latent, real_camera_keys  # noqa: E402
from common.runtime import build_runtime, normalize_single_visible_egl_device  # noqa: E402
from predictor.dataset import PERFRAME_FORMAT_VERSION, select_episode_row_mask  # noqa: E402


def shard_rows(episode_index, *, max_episodes: int, num_workers: int, worker_id: int) -> list[int]:
    """Row indices this worker collects: first ``max_episodes`` episodes, then this worker's share.

    Partitioning by ``episode_index % num_workers`` keeps whole episodes together, which is what
    the run-contiguity reconstruction at training time depends on.
    """
    episode_index = np.asarray(episode_index)
    rows = np.nonzero(select_episode_row_mask(episode_index, max_episodes))[0].tolist()
    if num_workers > 1:
        rows = [r for r in rows if int(episode_index[r]) % num_workers == worker_id]
    return rows


def write_shard(path, *, latents, actions, episode_index, frame_index, camera_keys) -> None:
    torch.save({
        "format_version": PERFRAME_FORMAT_VERSION,
        "latents": latents,
        "actions": actions,
        "episode_index": np.asarray(episode_index, dtype=np.int64),
        "frame_index": np.asarray(frame_index, dtype=np.int64),
        "camera_keys": tuple(camera_keys),
    }, path)


def collect(args) -> None:
    normalize_single_visible_egl_device()
    spec = get_backbone(args.backbone)
    runtime = build_runtime(
        spec.name, policy_path=args.policy_path, lerobot_root=args.lerobot_root,
        device=args.device, tokenizer_path=args.tokenizer_path,
        task_suite_name=args.task_suite_name, task_ids=args.task_ids, max_steps=args.max_steps,
        control_mode=args.control_mode)
    policy, pre = runtime.policy, runtime.preprocessor
    obs_state_key = runtime.obs_state_key
    policy.eval()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(args.repo_id, root=args.dataset_root, revision=args.dataset_revision)
    stats = load_action_stats(args.action_stats)

    cols = ds.hf_dataset.select_columns(
        ["episode_index", "frame_index", "action"]).with_format("numpy")
    episode_index = np.asarray(cols["episode_index"])
    frame_index = np.asarray(cols["frame_index"])
    actions_all = np.asarray(cols["action"], dtype=np.float32)   # ENV space

    rows = shard_rows(episode_index, max_episodes=args.max_episodes,
                      num_workers=args.num_workers, worker_id=args.worker_id)
    n_episodes = len({int(episode_index[r]) for r in rows})
    print(f"[worker {args.worker_id}/{args.num_workers}] {len(rows)}/{len(episode_index)} frames "
          f"from {n_episodes} episodes", flush=True)

    @torch.no_grad()
    def embed(items):
        obs = {
            "observation.images.image": torch.stack(
                [it["observation.images.image"] for it in items]).to(args.device),
            "observation.images.image2": torch.stack(
                [it["observation.images.image2"] for it in items]).to(args.device),
            obs_state_key: torch.stack(
                [it["observation.state"].float() for it in items]).to(args.device),
            "task": [it["task"] for it in items],
        }
        batch = pre(obs)
        return capture_latent(spec.name, policy, batch).cpu(), real_camera_keys(spec.name, policy)

    store_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[args.dtype]
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lat_buf, act_buf, ei_buf, fi_buf = [], [], [], []
    shard_i = args.worker_id * args.shard_index_stride
    camera_keys_out = None

    def flush():
        nonlocal shard_i, lat_buf, act_buf, ei_buf, fi_buf
        if not lat_buf:
            return
        write_shard(out / f"shard_{shard_i:06d}.pt",
                    latents=torch.cat(lat_buf, dim=0), actions=torch.stack(act_buf, dim=0),
                    episode_index=ei_buf, frame_index=fi_buf, camera_keys=camera_keys_out)
        print(f"wrote shard_{shard_i:06d}.pt ({len(lat_buf)} frames)", flush=True)
        shard_i += 1
        lat_buf, act_buf, ei_buf, fi_buf = [], [], [], []

    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        t0 = _time.perf_counter()
        items = [ds[g] for g in chunk]
        t1 = _time.perf_counter()
        z_batch, cam_keys = embed(items)
        if camera_keys_out is None:
            camera_keys_out = cam_keys
        t2 = _time.perf_counter()
        for j, g in enumerate(chunk):
            lat_buf.append(z_batch[j:j + 1].to(store_dtype))
            act_buf.append(torch.as_tensor(
                np.asarray(env_to_qnorm(actions_all[g], stats), dtype=np.float32)))
            ei_buf.append(int(episode_index[g]))
            fi_buf.append(int(frame_index[g]))
            if len(lat_buf) >= args.shard_frames:
                flush()
        print(f"  frames {min(i + args.batch_size, len(rows))}/{len(rows)} "
              f"decode={t1 - t0:.2f}s embed(b={len(chunk)})={t2 - t1:.2f}s", flush=True)
    flush()
    print("done", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", required=True, choices=["pi05", "smolvla"])
    p.add_argument("--policy-path", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--dataset-revision", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    p.add_argument("--lerobot-root", default=None,
                   help="LeRobot source checkout (only needed when LeRobot is not "
                        "installed in this environment); also read from LEROBOT_ROOT")
    p.add_argument("--tokenizer-path", default=None,
                   help="external PaliGemma tokenizer (pi0.5 only; SmolVLA keeps its own)")
    p.add_argument("--action-stats", default=None,
                   help="override assets/libero_action_stats.json")
    p.add_argument("--max-episodes", type=int, default=0, help="keep first N episodes (0=all)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--shard-frames", type=int, default=20000)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--shard-index-stride", type=int, default=100000)
    p.add_argument("--task-suite-name", default="libero_spatial")
    p.add_argument("--task-ids", default="0")
    p.add_argument("--control-mode", default="relative")
    p.add_argument("--max-steps", type=int, default=220)
    p.add_argument("--device", default="cuda")
    return p


if __name__ == "__main__":
    collect(build_parser().parse_args())
