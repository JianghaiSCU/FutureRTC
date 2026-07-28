#!/usr/bin/env python3
"""Train the full-state residual corrector on LIBERO demonstrations.

Three stages:

1. estimate a per-group (position / rotation / gripper) residual scale from a warmup sample of
   targets, so the three physically incomparable blocks contribute comparably to the loss;
2. train the MLP on normalized targets with plain MSE;
3. report held-out error against the true state -- position L2, rotation geodesic, gripper L2 --
   for the analytic proxy and for the corrected state, so the learned residual's contribution is
   visible on its own.

No policy, no VLM, no images: the corrector reads only the state and action columns.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corrector.dataset import ResidualDataset  # noqa: E402
from corrector.model import (  # noqa: E402
    RESIDUAL_TYPE, FullStateResidualMLP, save_corrector,
)
from corrector.physics import (  # noqa: E402
    FEATURE_DIM, apply_full_state_residual, axisangle_to_matrix,
    compute_group_residual_scale, denormalize_full_state_residual,
)


def _rot_geodesic(a_rotvec, b_rotvec):
    errs = []
    for a, b in zip(a_rotvec, b_rotvec):
        rel = axisangle_to_matrix(b) @ axisangle_to_matrix(a).T
        c = float(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
        errs.append(np.arccos(c))
    return np.asarray(errs, dtype=np.float32)


def _summ(pred, gt):
    pos = np.linalg.norm(pred[:, :3] - gt[:, :3], axis=1)
    rot = _rot_geodesic(pred[:, 3:6], gt[:, 3:6])
    grip = np.linalg.norm(pred[:, 6:8] - gt[:, 6:8], axis=1)
    return {"pos_l2_mean": float(pos.mean()), "pos_l2_p95": float(np.percentile(pos, 95)),
            "rot_geo_mean": float(rot.mean()), "rot_geo_p95": float(np.percentile(rot, 95)),
            "grip_l2_mean": float(grip.mean())}


def cosine_lr(step: int, *, lr: float, lr_min: float, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return lr * float(step + 1) / float(warmup_steps)
    span = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / span))
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * progress))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True, help="local LeRobot LIBERO dataset root")
    ap.add_argument("--dataset-revision", required=True)
    ap.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    ap.add_argument("--d-max", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=0.0)
    ap.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    ap.add_argument("--lr-warmup-steps", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=3000)
    ap.add_argument("--scale-warmup-batches", type=int, default=20)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--output", default="outputs/corrector/full_state_residual_d0_20.pt")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    base = LeRobotDataset(args.repo_id, root=args.dataset_root, revision=args.dataset_revision)
    ds = ResidualDataset.from_lerobot(base, d_max=args.d_max, seed=args.seed)
    print(f"dataset: {len(ds)} start-frames, d~U{{0..{args.d_max}}}", flush=True)

    def loader(seed):
        g = torch.Generator()
        g.manual_seed(seed)
        return torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
            drop_last=True, generator=g, persistent_workers=args.num_workers > 0)

    # 1) group residual scale from a warmup sample of targets
    targets = []
    for k, batch in enumerate(loader(args.seed)):
        targets.append(batch["target"].numpy())
        if k + 1 >= args.scale_warmup_batches:
            break
    residual_scale = compute_group_residual_scale(np.concatenate(targets, axis=0))
    print("residual_scale (pos/rot/grip std):", residual_scale.tolist(), flush=True)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = FullStateResidualMLP(input_dim=FEATURE_DIM, hidden_dim=args.hidden_dim,
                                 num_layers=args.num_layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scale_t = torch.from_numpy(residual_scale).to(dev)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    base_meta = {
        "residual_type": RESIDUAL_TYPE, "model_type": "mlp", "input_dim": FEATURE_DIM,
        "hidden_dim": args.hidden_dim, "num_layers": args.num_layers, "max_delay": args.d_max,
        "target_normalization": "group_scalar_std", "residual_scale": residual_scale.tolist(),
        "delays": f"U0..{args.d_max}", "seed": args.seed,
    }

    # 2) train
    step = 0
    metrics = []
    model.train()
    while step < args.steps:
        for batch in loader(args.seed + 1 + step):
            if args.lr_schedule == "cosine":
                lr = cosine_lr(step, lr=args.lr, lr_min=args.lr_min,
                               warmup_steps=args.lr_warmup_steps, total_steps=args.steps)
                for group in opt.param_groups:
                    group["lr"] = lr
            x = batch["features"].to(dev)
            y = batch["target"].to(dev) / scale_t
            loss = ((model(x) - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 500 == 0:
                metrics.append({"step": step, "loss": float(loss.item())})
                print(f"step {step} norm_mse {float(loss.item()):.6f}", flush=True)
            step += 1
            if args.ckpt_every and step % args.ckpt_every == 0 and step < args.steps:
                save_corrector(out, model, **{**base_meta, "step": step})
                print(f"[ckpt] saved intermediate at step {step} -> {out}", flush=True)
            if step >= args.steps:
                break

    # 3) held-out error vs the true state
    model.eval()
    proxies, corrected, truth = [], [], []
    with torch.no_grad():
        for k, batch in enumerate(loader(args.seed + 777)):
            pred_norm = model(batch["features"].to(dev)).cpu().numpy()
            residual = denormalize_full_state_residual(pred_norm, residual_scale)
            proxy, gt = batch["proxy"].numpy(), batch["gt"].numpy()
            proxies.append(proxy)
            corrected.append(apply_full_state_residual(proxy, residual))
            truth.append(gt)
            if k + 1 >= args.eval_batches:
                break
    proxies = np.concatenate(proxies)
    corrected = np.concatenate(corrected)
    truth = np.concatenate(truth)
    report = {"proxy_vs_gt": _summ(proxies, truth),
              "corrected_vs_gt": _summ(corrected, truth),
              "n_eval": int(len(truth))}
    print("=== vs-GT-state error ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)

    meta = {**base_meta, "steps": args.steps, "metrics_vs_gt": report}
    save_corrector(out, model.cpu(), **meta)
    out.with_suffix(".metrics.json").write_text(json.dumps({"train": metrics, **meta}, indent=2))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
