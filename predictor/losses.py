"""Reconstruction losses for the latent predictor.

Two terms, both reported unweighted so runs stay comparable across weight settings:

* ``mse``     -- per-camera mean squared error against the true handoff latent (constrains
                 magnitude).
* ``feature`` -- per-camera cosine distance (constrains direction). Optional: set
                 ``feature_weight=0`` to train on MSE alone.

The action-space policy-distillation term lives in ``predictor.policy_loss``; it needs
de-normalized latents and a loaded backbone, so the trainer adds it on top of this.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_losses(pred: torch.Tensor, z_target: torch.Tensor, *, mse_weight: float = 1.0,
                   feature_weight: float = 1.0) -> dict[str, torch.Tensor]:
    if pred.ndim != 4:
        raise ValueError(f"pred must have shape [B, C, T, D], got {tuple(pred.shape)}")
    if pred.shape != z_target.shape:
        raise ValueError(
            f"pred and z_target must have equal shapes, got {tuple(pred.shape)} and "
            f"{tuple(z_target.shape)}")

    per_camera_feature_loss = 1.0 - F.cosine_similarity(
        pred.flatten(2), z_target.flatten(2), dim=-1).mean(dim=0)
    per_camera_mse_loss = (pred - z_target).pow(2).mean(dim=(0, 2, 3))
    feature_loss = per_camera_feature_loss.mean()
    mse_loss = per_camera_mse_loss.mean()
    return {
        "loss": mse_weight * mse_loss + feature_weight * feature_loss,
        "mse_loss": mse_loss,
        "feature_loss": feature_loss,
        "per_camera_mse_loss": per_camera_mse_loss,
        "per_camera_feature_loss": per_camera_feature_loss,
    }
