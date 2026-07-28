#!/usr/bin/env python3
"""Per-(camera, hidden_dim) mean/std over a latent bank.

The trainer normalizes latents with these statistics and embeds them in the checkpoint, so the eval
driver can de-normalize a forecast without re-reading the bank. Statistics always come from the
FIRST bank of a multi-bank run, which keeps predictors comparable across runs.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictor.dataset import PerFrameLatentDataset, bank_shards, compute_latent_stats  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank-dir", required=True)
    p.add_argument("--output", default=None, help="default: <bank-dir>/latent_stats.pt")
    p.add_argument("--chunk", type=int, default=1024, help="frames per float64 accumulation chunk")
    args = p.parse_args()

    paths = bank_shards(args.bank_dir)
    dataset = PerFrameLatentDataset.from_shards(paths)
    print(f"{len(paths)} shards, {len(dataset.latents)} frames", flush=True)
    mean, std = compute_latent_stats(dataset.latents, chunk=args.chunk)
    out = pathlib.Path(args.output or (pathlib.Path(args.bank_dir) / "latent_stats.pt"))
    torch.save({"mean": mean, "std": std}, out)
    print(f"wrote {out}  mean{tuple(mean.shape)} std{tuple(std.shape)}", flush=True)


if __name__ == "__main__":
    main()
