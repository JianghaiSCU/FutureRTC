#!/usr/bin/env python3
"""State + task sidecar for a per-frame latent bank.

The latent bank stores only (latents, actions, episode_index, frame_index). Two other inputs the
pipeline needs are cheap and live in the SAME LeRobot dataset: the 8-dim proprioceptive state (the
corrector's anchor and the predictor's state channel) and the task instruction (what the flow head
is conditioned on during policy distillation). This extracts both, aligned to the bank's row
selection and order, without decoding a single image.

The alignment check against the bank is not optional in spirit: a silent (episode, frame) mismatch
would pair every state with the wrong latent.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictor.dataset import bank_shards, select_episode_row_mask  # noqa: E402


def task_index_to_strings(ds) -> list[str]:
    """``ds.meta.tasks`` is a DataFrame indexed by instruction with a 'task_index' column."""
    tasks = ds.meta.tasks
    idx_to_task = {int(ti): str(ts) for ts, ti in zip(tasks.index, tasks["task_index"])}
    n = max(idx_to_task) + 1
    return [idx_to_task.get(i, "") for i in range(n)]


def verify_against_bank(bank_dir, episode_index, frame_index) -> None:
    b_ei, b_fi = [], []
    for path in bank_shards(bank_dir):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        b_ei.append(np.asarray(payload["episode_index"]))
        b_fi.append(np.asarray(payload["frame_index"]))
    b_ei = np.concatenate(b_ei)
    b_fi = np.concatenate(b_fi)
    if (b_ei.shape != episode_index.shape or not np.array_equal(b_ei, episode_index)
            or not np.array_equal(b_fi, frame_index)):
        raise ValueError(
            f"sidecar (episode, frame) order does not match the bank: "
            f"sidecar N={episode_index.shape[0]} bank N={b_ei.shape[0]}")
    print(f"verified alignment vs bank: {episode_index.shape[0]} frames match", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--dataset-revision", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    p.add_argument("--max-episodes", type=int, default=0,
                   help="keep first N episodes (0=all); must match the bank's collection")
    p.add_argument("--bank-dir", default=None,
                   help="latent bank to verify (episode, frame) alignment against")
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(args.repo_id, root=args.dataset_root, revision=args.dataset_revision)
    hf = ds.hf_dataset.select_columns(
        ["observation.state", "task_index", "episode_index", "frame_index"]).with_format("numpy")
    cols = hf[:]
    episode_index = np.asarray(cols["episode_index"]).astype(np.int64)
    frame_index = np.asarray(cols["frame_index"]).astype(np.int64)
    state = np.asarray(cols["observation.state"], dtype=np.float32)
    task_index = np.asarray(cols["task_index"]).astype(np.int64)

    rows = np.nonzero(select_episode_row_mask(episode_index, args.max_episodes))[0]
    episode_index, frame_index = episode_index[rows], frame_index[rows]
    state, task_index = state[rows], task_index[rows]
    print(f"selected {len(rows)} frames; state dim {state.shape[1]}", flush=True)

    task_strings = task_index_to_strings(ds)
    if args.bank_dir:
        verify_against_bank(args.bank_dir, episode_index, frame_index)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": "state_task_sidecar_v1",
        "state": torch.from_numpy(state),
        "episode_index": torch.from_numpy(episode_index),
        "frame_index": torch.from_numpy(frame_index),
        "task_index": torch.from_numpy(task_index),
        "task_strings": task_strings,
    }, out)
    print(f"wrote {out} ({len(rows)} frames, {state.nbytes / 1e6:.1f} MB state)", flush=True)


if __name__ == "__main__":
    main()
