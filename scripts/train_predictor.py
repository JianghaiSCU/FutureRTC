#!/usr/bin/env python3
"""Stage 2 - train one motion-prior latent predictor per Kinetix level (MSE loss only).

Loss = mean squared error between the predicted environment latent and the collected
target latent:

    loss = mean( (predict_obs_latent(z_s, motion_actions, delay) - z_target)**2 )

That is the entire training objective. Because there is no policy / feature / regularization
term, training needs ONLY the ``.npz`` latent shards from Stage 1 - no RTC env or base
policy is loaded here.

Example (12 levels; --steps defaults to 15000):
  python scripts/train_predictor.py \
    --data-dir outputs/latents --output-dir outputs/predictors
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from motion_prior_handoff.flow_policy import level_name_from_path, parse_level_indices  # noqa: E402
from motion_prior_handoff.predictor import (  # noqa: E402
    FORMAT_VERSION,
    PREDICTOR_ARCHITECTURE,
    init_predictor_params,
    load_predictor_checkpoint,
    predict_obs_latent,
    save_predictor_checkpoint,
)
from motion_prior_handoff.rtc_env import DEFAULT_LEVEL_PATHS  # noqa: E402


def load_level_data(path: pathlib.Path):
    payload = np.load(path, allow_pickle=True)
    metadata = payload["metadata"].item()
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path} must use latent-shard format_version={FORMAT_VERSION}")
    arrays = {key: payload[key] for key in ("z_s", "z_target", "motion_actions", "delay")}
    if arrays["z_s"].ndim != 2 or arrays["z_target"].shape != arrays["z_s"].shape:
        raise ValueError(f"{path} must contain z_s/z_target with shape [N, D]")
    if arrays["motion_actions"].ndim != 3:
        raise ValueError(f"{path} must contain motion_actions with shape [N, H, A]")
    delays = arrays["delay"].astype(np.int64)
    if np.any(delays <= 0) or np.any(delays > arrays["motion_actions"].shape[1]):
        raise ValueError(f"{path} must contain positive delay within the action chunk")
    step_ids = np.arange(arrays["motion_actions"].shape[1])[None, :]
    padded_region = step_ids >= delays[:, None]
    if np.any(arrays["motion_actions"][padded_region] != 0):
        raise ValueError(f"{path} motion_actions must be zero-padded after the executed delay actions")
    return arrays, metadata


def train_level(args, *, data, metadata, output_path: pathlib.Path):
    import jax
    import jax.numpy as jnp
    import optax

    latent_dim = int(metadata["latent_dim"])
    action_dim = int(metadata["action_dim"])
    action_chunk_size = int(metadata["action_chunk_size"])
    max_delay = max(int(value) for value in metadata["delays"])
    if data["motion_actions"].shape[1:] != (action_chunk_size, action_dim):
        raise ValueError("motion_actions shape does not match shard metadata")
    if data["z_s"].shape[1] != latent_dim:
        raise ValueError("z_s shape does not match shard metadata")

    rng = jax.random.key(args.seed)
    rng, key = jax.random.split(rng)
    params = init_predictor_params(
        key,
        latent_dim=latent_dim,
        action_dim=action_dim,
        action_chunk_size=action_chunk_size,
        max_delay=max_delay,
        hidden_dim=args.hidden_dim,
        num_streams=1,
        action_num_heads=args.action_num_heads,
        action_num_layers=args.action_num_layers,
    )
    if args.init_from:
        # Continue a previous block (cross-process resume); optimizer state restarts.
        init_path = pathlib.Path(args.init_from) / f"{metadata['level_name']}.pkl"
        params, _init_md = load_predictor_checkpoint(init_path)
        print(f"  init params from {init_path}", flush=True)
    optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    def loss_fn(params, batch):
        pred = predict_obs_latent(
            params, batch["z_s"], batch["motion_actions"], batch["delay"], max_delay=max_delay
        )
        return jnp.mean(jnp.square(pred - batch["z_target"]))

    @jax.jit
    def train_step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    num_samples = data["z_s"].shape[0]
    if num_samples == 0:
        raise ValueError("cannot train on an empty latent shard")

    def make_md(total_steps):
        return {
            **metadata,
            "hidden_dim": args.hidden_dim,
            "num_streams": 1,
            "predictor_architecture": PREDICTOR_ARCHITECTURE,
            "max_delay": max_delay,
            "action_num_heads": args.action_num_heads,
            "action_num_layers": args.action_num_layers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "loss": "mse",
            "steps": total_steps,
        }

    base_step = int(args.base_step)
    ckpt_root = pathlib.Path(args.checkpoint_root) if args.checkpoint_root else output_path.parent.parent

    metrics_rows = []
    for step in range(args.steps):
        rng, index_key = jax.random.split(rng)
        indices = jax.random.randint(index_key, (args.batch_size,), 0, num_samples)
        batch = {key: jnp.asarray(value)[indices] for key, value in data.items()}
        params, opt_state, loss = train_step(params, opt_state, batch)
        metrics_rows.append({"mse_loss": float(loss)})
        if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0:
            total = base_step + step + 1
            snap_dir = ckpt_root / f"step{total}"
            snap_dir.mkdir(parents=True, exist_ok=True)
            save_predictor_checkpoint(
                snap_dir / f"{metadata['level_name']}.pkl", params=params, metadata=make_md(total)
            )

    save_predictor_checkpoint(output_path, params=params, metadata=make_md(base_step + args.steps))
    output_path.with_suffix(".metrics.json").write_text(json.dumps(metrics_rows, indent=2))


def train(args) -> None:
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    level_indices = parse_level_indices(args.level_indices, len(DEFAULT_LEVEL_PATHS))

    for level_index in level_indices:
        level_name = level_name_from_path(DEFAULT_LEVEL_PATHS[level_index])
        data_path = pathlib.Path(args.data_dir) / f"{level_name}.npz"
        data, metadata = load_level_data(data_path)
        train_level(args, data=data, metadata=metadata, output_path=output_dir / f"{level_name}.pkl")
        print(level_name, "done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level-indices", default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--data-dir", default="outputs/latents")
    parser.add_argument("--output-dir", default="outputs/predictors")
    parser.add_argument("--init-from", default=None, help="dir with {level}.pkl to continue from")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="save a snapshot every N steps into {checkpoint-root}/step{base+N}/ (0=off)")
    parser.add_argument("--base-step", type=int, default=0,
                        help="step count already trained in --init-from, for cumulative snapshot naming")
    parser.add_argument("--checkpoint-root", default=None,
                        help="root for periodic snapshots (default: parent of --output-dir)")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--action-num-heads", type=int, default=4)
    parser.add_argument("--action-num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
