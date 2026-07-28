"""Action-space policy-distillation loss.

Runs the FROZEN backbone flow head on the predicted latent and on the true target latent with the
same proprioceptive state, task, and flow noise, then matches the single-step action estimates.
Gradients reach only the predicted latent, so the predictor is pushed toward latents the policy
turns into the right ACTION -- a downstream-aligned signal that latent MSE is a poor proxy for.

Inputs come from the per-frame latent bank plus the small state/task sidecar, so no images are
read and the latent bank is never re-embedded.
"""
from __future__ import annotations

from common.backbones import get_backbone
from predictor.policy_loss.base import OfflinePolicyLoss, build_x_t


def build_policy_loss(backbone: str, **kwargs) -> OfflinePolicyLoss:
    spec = get_backbone(backbone)
    if spec.name == "pi05":
        from predictor.policy_loss.pi05 import OfflinePi05PolicyLoss
        return OfflinePi05PolicyLoss(**kwargs)
    from predictor.policy_loss.smolvla import OfflineSmolvlaPolicyLoss
    return OfflineSmolvlaPolicyLoss(**kwargs)


__all__ = ["OfflinePolicyLoss", "build_policy_loss", "build_x_t"]
