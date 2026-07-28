"""Capture and injection of the backbone's real-camera visual tokens.

This is the entire model-dispatch surface of the bundle. Three callers depend on it:

* ``predictor.collect_latents``  -- builds the latent bank,
* ``predictor.policy_loss``      -- runs the frozen flow head on a predicted latent,
* ``eval.driver``                -- forecasts and injects at handoff time.

Because all three go through the same boundary, the bank the predictor trains on and the tokens the
deployed policy consumes are identical by construction.

The boundary differs per backbone:

* pi0.5 concatenates every image's tokens into one prefix before the language suffix, so the real
  camera spans are reconstructed from the image masks (``VisualPrefixLayout``) and read out of the
  concatenated tensor.
* SmolVLA calls ``vlm_with_expert.embed_image`` once per camera, which is layout-agnostic, so the
  per-camera outputs are captured directly.
"""
from __future__ import annotations

import dataclasses
import types

import torch

from common.backbones import get_backbone


# --------------------------------------------------------------------------------------
# pi0.5: prefix layout
# --------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class VisualPrefixLayout:
    """Token layout for image slots before the language suffix in a pi0.5 prefix."""

    image_slot_count: int
    real_slot_indices: tuple[int, ...]
    tokens_per_image: int
    visual_token_count: int
    language_token_count: int


def get_embed_prefix_owner(policy):
    try:
        return policy.model
    except AttributeError as exc:
        raise AttributeError(
            "expected policy.model.embed_prefix; this helper is specific to LeRobot PI05Policy"
        ) from exc


def get_real_camera_keys(policy) -> tuple[str, ...]:
    try:
        image_features = policy.config.image_features
    except AttributeError as exc:
        raise AttributeError("expected policy.config.image_features for the camera layout") from exc
    return tuple(key for key in image_features if "empty_camera" not in key)


def infer_visual_prefix_layout(
    prefix_embs: torch.Tensor,
    img_masks: list[torch.Tensor],
    tokens: torch.Tensor,
) -> VisualPrefixLayout:
    """Infer equal-sized image spans and identify fully-present real camera slots."""

    if prefix_embs.ndim != 3:
        raise ValueError(f"prefix_embs must have shape [B, P, D], got {tuple(prefix_embs.shape)}")
    if tokens.ndim < 2:
        raise ValueError(f"tokens must have shape [B, L], got {tuple(tokens.shape)}")
    if not img_masks:
        raise ValueError("img_masks must contain at least one image slot")

    language_token_count = int(tokens.shape[1])
    visual_token_count = int(prefix_embs.shape[1] - language_token_count)
    image_slot_count = len(img_masks)
    if visual_token_count <= 0 or visual_token_count % image_slot_count:
        raise ValueError(
            "cannot split visual prefix into equal image spans: "
            f"prefix_tokens={prefix_embs.shape[1]} language_tokens={language_token_count} "
            f"image_slots={image_slot_count}"
        )

    real_slot_indices = []
    for slot, mask in enumerate(img_masks):
        if mask.ndim != 1 or mask.shape[0] != prefix_embs.shape[0]:
            raise ValueError(
                f"img_masks[{slot}] must have shape [B], got {tuple(mask.shape)} "
                f"for batch size {prefix_embs.shape[0]}"
            )
        if bool(mask.all()):
            real_slot_indices.append(slot)
        elif bool(mask.any()):
            raise ValueError(
                f"img_masks[{slot}] mixes real and empty cameras within one batch; "
                "prefix capture requires a stable camera layout"
            )
    if not real_slot_indices:
        raise ValueError("no real camera slots found in img_masks")

    return VisualPrefixLayout(
        image_slot_count=image_slot_count,
        real_slot_indices=tuple(real_slot_indices),
        tokens_per_image=visual_token_count // image_slot_count,
        visual_token_count=visual_token_count,
        language_token_count=language_token_count,
    )


def extract_real_visual_tokens(prefix_embs: torch.Tensor,
                               layout: VisualPrefixLayout) -> torch.Tensor:
    """Return real-camera tokens as [B, C, T, D], preserving slot order."""

    spans = []
    for slot in layout.real_slot_indices:
        start = slot * layout.tokens_per_image
        end = start + layout.tokens_per_image
        spans.append(prefix_embs[:, start:end])
    return torch.stack(spans, dim=1)


def replace_real_visual_tokens(
    prefix_embs: torch.Tensor,
    layout: VisualPrefixLayout,
    replacement: torch.Tensor,
) -> torch.Tensor:
    """Write [B, C, T, D] real-camera tokens back into a full prefix."""

    expected = (
        prefix_embs.shape[0],
        len(layout.real_slot_indices),
        layout.tokens_per_image,
        prefix_embs.shape[-1],
    )
    if tuple(replacement.shape) != expected:
        raise ValueError(f"replacement must have shape {expected}, got {tuple(replacement.shape)}")
    output = prefix_embs.clone()
    for camera_index, slot in enumerate(layout.real_slot_indices):
        start = slot * layout.tokens_per_image
        end = start + layout.tokens_per_image
        output[:, start:end] = replacement[:, camera_index]
    return output


class EmbedPrefixCapture:
    """Capture all real-camera visual tokens after pi0.5 prefix concatenation."""

    def __init__(self, policy):
        self.policy = policy
        self.owner = get_embed_prefix_owner(policy)
        self.original = self.owner.embed_prefix
        self.had_instance_attr = "embed_prefix" in getattr(self.owner, "__dict__", {})
        self.value = None
        self.layout = None
        self.camera_keys = None
        self.seen = False

    def _patched(self, owner, images, img_masks, tokens, masks):
        if self.seen:
            raise RuntimeError("embed_prefix was called more than once during one capture context")
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.original(
            images, img_masks, tokens, masks)
        layout = infer_visual_prefix_layout(prefix_embs, img_masks, tokens)
        camera_keys = get_real_camera_keys(self.policy)
        if len(camera_keys) != len(layout.real_slot_indices):
            raise RuntimeError(
                "pi0.5 real camera configuration does not match runtime image masks: "
                f"camera_keys={camera_keys} real_slots={layout.real_slot_indices}"
            )
        self.value = extract_real_visual_tokens(prefix_embs, layout).detach().cpu()
        self.layout = layout
        self.camera_keys = camera_keys
        self.seen = True
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def __enter__(self):
        self.owner.embed_prefix = types.MethodType(self._patched, self.owner)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.had_instance_attr:
            self.owner.embed_prefix = self.original
        else:
            delattr(self.owner, "embed_prefix")
        return False


class InjectVisualPrefix:
    """Patch ``embed_prefix`` to overwrite the real-camera tokens with a provided ``z`` [B,C,T,D].

    The latent is injected as-is, so the caller controls the autograd graph into ``z``.
    """

    def __init__(self, policy, z: torch.Tensor):
        self.owner = get_embed_prefix_owner(policy)
        self.original = self.owner.embed_prefix
        self.z = z
        self.had_instance_attr = "embed_prefix" in getattr(self.owner, "__dict__", {})
        self.seen = False

    def _patched(self, owner, images, img_masks, tokens, masks):
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.original(
            images, img_masks, tokens, masks)
        layout = infer_visual_prefix_layout(prefix_embs, img_masks, tokens)
        prefix_embs = replace_real_visual_tokens(prefix_embs, layout,
                                                 self.z.to(prefix_embs.dtype))
        self.seen = True
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def __enter__(self):
        self.owner.embed_prefix = types.MethodType(self._patched, self.owner)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.had_instance_attr:
            self.owner.embed_prefix = self.original
        else:
            delattr(self.owner, "embed_prefix")
        return False


# --------------------------------------------------------------------------------------
# SmolVLA: per-camera embed_image
# --------------------------------------------------------------------------------------

class SmolvlaEmbedImageCapture:
    """Patch ``vlm_with_expert.embed_image`` to record each per-camera output; stack -> [B,C,T,D]."""

    def __init__(self, model):
        self.owner = model.vlm_with_expert
        self.original = self.owner.embed_image
        self.captured: list[torch.Tensor] = []
        self.had_instance_attr = "embed_image" in getattr(self.owner, "__dict__", {})

        def patched(owner, img):
            out = self.original(img)
            self.captured.append(out.detach())
            return out

        self._patched = types.MethodType(patched, self.owner)

    def install(self):
        self.captured = []
        self.owner.embed_image = self._patched

    def restore(self):
        if self.had_instance_attr:
            self.owner.embed_image = self.original
        else:
            self.owner.__dict__.pop("embed_image", None)

    def stack(self) -> torch.Tensor:
        if not self.captured:
            raise RuntimeError("embed_image captured nothing (embed_prefix not run?)")
        return torch.stack(self.captured, dim=1)  # [B, C, T, D]

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *a):
        self.restore()
        return False


class SmolvlaInjectImage:
    """Patch ``embed_image`` to return the provided ``z`` [B,C,T,D] per camera (grad-friendly)."""

    def __init__(self, model, z: torch.Tensor):
        self.owner = model.vlm_with_expert
        self.original = self.owner.embed_image
        self.z = z
        self.i = 0
        self.had_instance_attr = "embed_image" in getattr(self.owner, "__dict__", {})

        def patched(owner, img):
            zi = self.z[:, self.i]
            self.i += 1
            return zi

        self._patched = types.MethodType(patched, self.owner)

    def __enter__(self):
        self.i = 0
        self.owner.embed_image = self._patched
        return self

    def __exit__(self, *a):
        if self.had_instance_attr:
            self.owner.embed_image = self.original
        else:
            self.owner.__dict__.pop("embed_image", None)
        return False


# --------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------

def real_camera_keys(backbone: str, policy) -> tuple[str, ...]:
    get_backbone(backbone)
    return get_real_camera_keys(policy)


def build_policy_inputs(backbone: str, policy, batch: dict):
    """Backbone-specific policy inputs derived from a preprocessed batch.

    pi0.5 -> ``(images, img_masks, tokens, masks)``: proprio state enters as discretized prompt
    tokens the preprocessor already baked into ``tokens``.
    SmolVLA -> ``(images, img_masks, lang_tokens, lang_masks, state)``: proprio state is a separate
    projected token passed straight to ``embed_prefix``.
    """
    spec = get_backbone(backbone)
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    if spec.name == "pi05":
        images, img_masks = policy._preprocess_images(batch)
        return images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    return (images, img_masks, batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK], state)


def capture_latent(backbone: str, policy, batch: dict) -> torch.Tensor:
    """Capture the real-camera visual tokens at the backbone's latent boundary -> [B, C, T, D].

    no_grad is load-bearing: capture is pure inference, and pi0.5's PaliGemma vision tower would
    otherwise build a full autograd graph and exhaust a 24 GB card.
    """
    spec = get_backbone(backbone)
    device = next(policy.model.parameters()).device
    with torch.no_grad():
        if spec.name == "pi05":
            images, img_masks, tokens, masks = build_policy_inputs(spec.name, policy, batch)
            with EmbedPrefixCapture(policy) as cap:
                policy.model.embed_prefix(images, img_masks, tokens, masks)
            z = cap.value
        else:
            images, img_masks, lang_tokens, lang_masks, state = build_policy_inputs(
                spec.name, policy, batch)
            with SmolvlaEmbedImageCapture(policy.model) as cap:
                policy.model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
            z = cap.stack()
    if z.shape[1] != spec.latent_cameras:
        raise RuntimeError(
            f"expected {spec.latent_cameras} real cameras for {spec.name}, captured {z.shape[1]}")
    return z.to(device)


def inject_latent(backbone: str, policy, z: torch.Tensor):
    """Context manager overwriting the real-camera visual tokens with ``z`` [B, C, T, D] for the
    enclosed policy forward. Grad-transparent, so callers control the graph into ``z``."""
    spec = get_backbone(backbone)
    if spec.name == "pi05":
        return InjectVisualPrefix(policy, z)
    return SmolvlaInjectImage(policy.model, z)
