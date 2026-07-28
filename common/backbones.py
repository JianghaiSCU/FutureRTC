"""The two supported backbones and their LIBERO wiring."""
from __future__ import annotations

import dataclasses
import importlib

_SMOLVLA_RENAME = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


@dataclasses.dataclass(frozen=True)
class BackboneSpec:
    name: str
    config_modules: tuple[str, ...]
    # LeRobot LIBERO env image keys -> the policy's expected keys. pi0.5 consumes image/image2
    # directly (None = no rename); SmolVLA was trained with a shipped rename processor that we
    # must replay so the env observations match the policy.
    rename_map: dict[str, str] | None
    # pi0.5 takes an external PaliGemma tokenizer; SmolVLA ships its own SmolVLM tokenizer and
    # must NOT be overridden.
    external_tokenizer: bool
    latent_dim: int
    latent_cameras: int
    tokens_per_camera: int


BACKBONES: dict[str, BackboneSpec] = {
    "pi05": BackboneSpec(
        name="pi05",
        config_modules=("lerobot.policies.pi05.configuration_pi05",),
        rename_map=None,
        external_tokenizer=True,
        latent_dim=2048,
        latent_cameras=2,
        tokens_per_camera=64,
    ),
    "smolvla": BackboneSpec(
        name="smolvla",
        config_modules=("lerobot.policies.smolvla.configuration_smolvla",),
        rename_map=dict(_SMOLVLA_RENAME),
        external_tokenizer=False,
        latent_dim=960,
        latent_cameras=2,
        tokens_per_camera=64,
    ),
}


def get_backbone(name: str) -> BackboneSpec:
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; known: {sorted(BACKBONES)}")
    spec = BACKBONES[name]
    return dataclasses.replace(
        spec, rename_map=None if spec.rename_map is None else dict(spec.rename_map))


def register_all_configs() -> None:
    """Import every backbone's config module so LeRobot can decode any checkpoint's config.json
    regardless of which backbone is selected. A missing optional dependency is not fatal."""
    for spec in BACKBONES.values():
        for module in spec.config_modules:
            try:
                importlib.import_module(module)
            except Exception:  # noqa: BLE001 - a backbone's optional deps may be absent
                pass
