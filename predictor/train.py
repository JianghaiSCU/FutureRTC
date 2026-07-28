#!/usr/bin/env python3
"""Train the visual-latent predictor.

Two phases, both driven by this script:

* phase 1 -- reconstruction: ``--mse-weight 1 --feature-weight 1 --policy-weight 0``
* phase 2 -- policy distillation: ``--resume-from <phase-1 ckpt> --policy-weight 10``

The loss is ``mse_weight * mse + feature_weight * feature + policy_weight * policy``. The policy
term needs de-normalized latents and a loaded backbone, so it is applied here rather than inside
``predictor.losses``.

The proprioceptive channel is the deployable one: the corrector's estimate of the handoff state,
computed in the DataLoader workers from the state ``d`` frames earlier plus the committed actions.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.backbones import get_backbone  # noqa: E402
from corrector.state_fn import CorrectorStateFn  # noqa: E402
from predictor.dataset import (  # noqa: E402
    PerFrameLatentDataset, bank_shards, collate, compute_latent_stats,
)
from predictor.losses import compute_losses  # noqa: E402
from predictor.model import (  # noqa: E402
    PREDICTOR_ARCHITECTURE, PREDICTOR_FORMAT_VERSION, MotionPriorLatentPredictor,
)


def cosine_lr(step: int, *, lr: float, lr_min: float, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay from ``lr`` to ``lr_min`` over the remaining steps."""
    if warmup_steps > 0 and step < warmup_steps:
        return lr * float(step + 1) / float(warmup_steps)
    span = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / span))
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * progress))


def build_mix_sampler(dataset, bank_frames, mix_weights):
    """Sample so each bank's EXPECTED share of the batch equals its weight, regardless of size.

    Without this the index is uniform over pairs, so a large bank simply drowns a small one.
    """
    if mix_weights is None or len(bank_frames) < 2:
        return None
    weights = [float(x) for x in mix_weights]
    if len(weights) != len(bank_frames):
        raise ValueError(
            f"need one mix weight per bank: got {len(weights)} for {len(bank_frames)} banks")
    bounds = torch.tensor(bank_frames).cumsum(0)
    g0 = torch.tensor([run_start + local_t for _rid, local_t, run_start in dataset.index],
                      dtype=torch.long)
    src = torch.bucketize(g0, bounds, right=True)
    counts = torch.bincount(src, minlength=len(bank_frames)).clamp_min(1)
    per_source = torch.tensor(weights) / counts.float()
    return torch.utils.data.WeightedRandomSampler(
        per_source[src].double(), num_samples=len(dataset), replacement=True)


def load_sidecars(paths, bank_frames):
    """Concatenate per-bank sidecars in bank order (the bank's global frame order is bank-major)."""
    if len(paths) != len(bank_frames):
        raise ValueError(
            f"need one sidecar per bank: got {len(paths)} for {len(bank_frames)} banks")
    states = []
    for path, n in zip(paths, bank_frames):
        sidecar = torch.load(path, map_location="cpu", weights_only=False)
        state = sidecar["state"].float()
        if state.shape[0] != n:
            raise ValueError(f"{path}: sidecar has {state.shape[0]} frames but its bank has {n}")
        states.append(state)
    return torch.cat(states, 0)


def build_bank(data_dirs, *, d_max, seed, state_fn=None):
    """Concatenate one or more latent banks; also returns each bank's frame count (bank-major)."""
    per_bank_paths = [bank_shards(d) for d in data_dirs]
    if len(data_dirs) > 1:
        print(f"multi-bank training: "
              f"{[f'{d} ({len(g)} shards)' for d, g in zip(data_dirs, per_bank_paths)]}", flush=True)
    paths = [p for group in per_bank_paths for p in group]
    dataset = PerFrameLatentDataset.from_shards(paths, d_max=d_max, seed=seed, state_fn=state_fn)
    if len(data_dirs) == 1:
        bank_frames = [dataset.actions.shape[0]]
    else:
        bank_frames = [
            sum(int(torch.load(p, map_location="cpu",
                               weights_only=False)["episode_index"].shape[0]) for p in group)
            for group in per_bank_paths
        ]
        if sum(bank_frames) != dataset.actions.shape[0]:
            raise RuntimeError(f"bank frame counts {bank_frames} sum to {sum(bank_frames)} != "
                               f"dataset {dataset.actions.shape[0]}")
    return dataset, bank_frames


def save_checkpoint(path, model, args, camera_keys, latent_norm) -> None:
    torch.save({
        "format_version": PREDICTOR_FORMAT_VERSION,
        "predictor_architecture": PREDICTOR_ARCHITECTURE,
        "model": model.state_dict(),
        "args": vars(args),
        "camera_keys": camera_keys,
        "latent_norm": latent_norm,
    }, path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", required=True, choices=["pi05", "smolvla"])
    p.add_argument("--data-dir", required=True,
                   help="latent bank dir, or a comma-separated list of banks to concatenate")
    p.add_argument("--data-mix-weight", default=None,
                   help="with multiple banks: comma-separated sampling weights, one per bank "
                        "(e.g. '1,1' = half the batch from each regardless of size). "
                        "Default: sample uniformly over all pairs (size-proportional).")
    p.add_argument("--state-sidecar", required=True,
                   help="state/task sidecar(s), one per bank in the same order")
    p.add_argument("--sidecar", default=None,
                   help="sidecar(s) for the policy loss; defaults to --state-sidecar")
    p.add_argument("--corrector-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-delay", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8,
                   help="DataLoader workers. Needs a SHORT TMPDIR: a long scratch path overflows "
                        "the ~108-character unix-socket limit and crashes worker startup.")
    p.add_argument("--steps", type=int, default=300000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    p.add_argument("--lr-warmup-steps", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mse-weight", type=float, default=1.0)
    p.add_argument("--feature-weight", type=float, default=1.0,
                   help="cosine term; set 0 to train on MSE alone")
    p.add_argument("--policy-weight", type=float, default=0.0,
                   help="action-space distillation weight; phase 2 uses 10.0")
    p.add_argument("--policy-path", default=None, help="backbone checkpoint (policy loss only)")
    p.add_argument("--lerobot-root", default=None,
                   help="LeRobot source checkout (only needed when LeRobot is not "
                        "installed in this environment); also read from LEROBOT_ROOT")
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--metrics-every", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=10000)
    p.add_argument("--release-resident-every", type=int, default=500,
                   help="drop resident mmap pages every N steps to bound RSS (0 disables)")
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--start-step", type=int, default=0)
    return p


def _endless(loader):
    # Re-iterate the DataLoader each epoch. itertools.cycle() would cache an entire epoch of
    # batches to replay it -- hundreds of GB of materialized float batches for this bank.
    while True:
        yield from loader


def train(args) -> None:
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    spec = get_backbone(args.backbone)

    data_dirs = [d.strip() for d in str(args.data_dir).split(",") if d.strip()]
    sidecar_paths = [s.strip() for s in str(args.state_sidecar).split(",") if s.strip()]
    mix_weights = ([w.strip() for w in str(args.data_mix_weight).split(",") if w.strip()]
                   if args.data_mix_weight else None)

    dataset, bank_frames = build_bank(data_dirs, d_max=args.max_delay, seed=args.seed)
    sidecar_state = load_sidecars(sidecar_paths, bank_frames)
    dataset.state_fn = CorrectorStateFn(
        args.corrector_checkpoint, sidecar_state.numpy(), d_max=args.max_delay,
        action_space="qnorm")
    print(f"corrector state channel ON (ckpt={args.corrector_checkpoint}, "
          f"{sidecar_state.shape[0]} sidecar frames)", flush=True)

    # Normalization always comes from the FIRST bank: every bank holds latents from the same
    # frozen VLM, and reusing one set of stats keeps predictors comparable across runs.
    stats_path = pathlib.Path(data_dirs[0]) / "latent_stats.pt"
    if stats_path.exists():
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        mean, std = stats["mean"], stats["std"]
        print(f"loaded latent stats from {stats_path}", flush=True)
    else:
        print("computing per-(camera, dim) latent mean/std over the bank ...", flush=True)
        mean, std = compute_latent_stats(dataset.latents)
        torch.save({"mean": mean, "std": std}, stats_path)
        print(f"saved latent stats to {stats_path}", flush=True)
    dataset.set_normalization(mean, std)
    latent_norm = {"mean": mean, "std": std}

    sample = dataset[0]
    num_streams, tokens_per_stream, latent_dim = sample["z_s"].shape
    if latent_dim != spec.latent_dim:
        raise ValueError(f"bank latent_dim {latent_dim} does not match backbone "
                         f"{spec.name} ({spec.latent_dim})")
    if num_streams != spec.latent_cameras:
        raise ValueError(f"bank has {num_streams} cameras, backbone {spec.name} expects "
                         f"{spec.latent_cameras}")
    args.latent_dim = latent_dim
    args.num_streams = num_streams
    args.tokens_per_stream = tokens_per_stream

    def _worker_init(worker_id):
        # Re-seed the delay-sampling RNG per worker; it is forked with an identical seed, so
        # without this every worker draws the same delay sequence.
        info = torch.utils.data.get_worker_info()
        if info is not None and hasattr(info.dataset, "_rng"):
            info.dataset._rng = random.Random(args.seed * 100003 + worker_id + 1)

    sampler = build_mix_sampler(dataset, bank_frames, mix_weights)
    if sampler is not None:
        total = sum(float(w) for w in mix_weights)
        print(f"bank mix: expected shares "
              f"{[round(float(w) / total * 100, 1) for w in mix_weights]}%", flush=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        collate_fn=collate, drop_last=True, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=str(args.device).startswith("cuda"),
        worker_init_fn=_worker_init if args.num_workers > 0 else None)

    model = MotionPriorLatentPredictor(latent_dim).to(args.device)
    print(f"predictor: latent_dim={latent_dim} "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    policy_loss_fn = None
    if args.policy_weight > 0:
        from corrector.state_fn import Corrector
        from predictor.policy_loss import build_policy_loss
        if not args.policy_path:
            raise ValueError("--policy-weight > 0 requires --policy-path")
        pl_sidecars = [s.strip() for s in str(args.sidecar or args.state_sidecar).split(",")
                       if s.strip()]
        policy_loss_fn = build_policy_loss(
            args.backbone, policy_path=args.policy_path, sidecar_paths=pl_sidecars,
            dataset_actions=dataset.actions, corrector=Corrector(args.corrector_checkpoint),
            device=args.device, lerobot_root=args.lerobot_root,
            tokenizer_path=args.tokenizer_path, seed=args.seed, max_delay=args.max_delay)
        print(f"policy distillation ON (weight {args.policy_weight})", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        start_step = int(args.start_step)
        # Optimizer state is not checkpointed; AdamW re-warms within a few hundred steps, which is
        # negligible over the remaining budget.
        print(f"resumed weights from {args.resume_from} at step {start_step}", flush=True)

    metrics = []
    metrics_path = output_dir / "metrics.json"
    if start_step > 0 and metrics_path.exists():
        try:  # keep the loss curve across a resume
            metrics = [m for m in json.loads(metrics_path.read_text()) if m["step"] <= start_step]
        except Exception:  # noqa: BLE001 - a corrupt curve must not block training
            metrics = []

    camera_keys = sample["camera_keys"]
    batches = _endless(loader)
    model.train()
    for step in range(start_step, args.steps):
        batch = next(batches)
        if args.lr_schedule == "cosine":
            lr = cosine_lr(step, lr=args.lr, lr_min=args.lr_min,
                           warmup_steps=args.lr_warmup_steps, total_steps=args.steps)
            for group in optim.param_groups:
                group["lr"] = lr
        else:
            lr = args.lr

        z_s = batch["z_s"].to(args.device, non_blocking=True)
        z_target = batch["z_target"].to(args.device, non_blocking=True)
        z_init = batch["z_init"].to(args.device, non_blocking=True)
        delay = batch["delay"].to(args.device, non_blocking=True)
        actions = batch["motion_actions"].to(args.device, non_blocking=True)
        state = batch["state"].to(args.device, non_blocking=True)

        pred = model(z_s, actions, delay, z_init=z_init, state=state)
        losses = compute_losses(pred, z_target, mse_weight=args.mse_weight,
                                feature_weight=args.feature_weight)
        loss = losses["loss"]

        policy_loss_val = 0.0
        if policy_loss_fn is not None:
            # the flow head expects RAW latents: de-normalize before distillation.
            m = latent_norm["mean"].to(pred.device)[None, :, None, :]
            s = latent_norm["std"].to(pred.device)[None, :, None, :]
            policy_loss = policy_loss_fn.loss(
                pred * s + m, z_target * s + m, batch["g_target"],
                committed_actions=batch["motion_actions"], delay=batch["delay"])
            loss = loss + args.policy_weight * policy_loss
            policy_loss_val = float(policy_loss.detach().cpu())

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        step_num = step + 1
        if step_num % args.metrics_every == 0 or step_num == args.steps:
            row = {
                "step": step_num,
                "lr": float(lr),
                "loss": float(loss.detach().cpu()),
                "mse_loss": float(losses["mse_loss"].detach().cpu()),
                "feature_loss": float(losses["feature_loss"].detach().cpu()),
                "policy_loss": policy_loss_val,
                "per_camera_mse_loss": {
                    k: float(v) for k, v in
                    zip(camera_keys, losses["per_camera_mse_loss"].detach().cpu().tolist())},
                "per_camera_feature_loss": {
                    k: float(v) for k, v in
                    zip(camera_keys, losses["per_camera_feature_loss"].detach().cpu().tolist())},
            }
            metrics.append(row)
            metrics_path.write_text(json.dumps(metrics, indent=2))
            print(f"step {step_num}/{args.steps} loss={row['loss']:.4f} "
                  f"mse={row['mse_loss']:.4f} feat={row['feature_loss']:.4f} "
                  f"policy={policy_loss_val:.4f}", flush=True)
        if args.ckpt_every and step_num % args.ckpt_every == 0 and step_num != args.steps:
            save_checkpoint(output_dir / f"predictor_step{step_num}.pt", model, args,
                            camera_keys, latent_norm)
        if (args.release_resident_every > 0 and step_num % args.release_resident_every == 0):
            dataset.release_resident()  # bound RSS on a mmap-backed bank

    save_checkpoint(output_dir / "predictor.pt", model, args, camera_keys, latent_norm)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"saved {output_dir / 'predictor.pt'}", flush=True)


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
