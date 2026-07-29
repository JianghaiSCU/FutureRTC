#!/usr/bin/env python3
"""Single-token motion-prior latent predictor for Kinetix (JAX).

Given the current full observation-latent ``z_s``, the executed action prefix
``motion_actions`` (the ``d`` actions run during the inference delay, zero-padded to the
chunk length), and the delay ``d``, this network predicts the future *environment*
observation-latent ``z_env(obs_{s+d})``. One lightweight predictor is trained per Kinetix
level; it is the learned component of FutureRTC on Kinetix.

Kinetix symbolic observations map to a single latent vector (one stream ``S=1``, one token
``T=1``), so the spatial machinery of the original image-token-grid predictor collapses to
identity and is omitted. What remains, acting at ``T=1``:

  - an action-trajectory encoder (temporal self-attention over the executed actions),
  - a per-token residual trunk,
  - a per-token transport gain head:  ``z_transport = z * (1 + strength * (gain - 1))``,
  - a gated residual (innovation) decoder:  ``z_hat = z_transport + gate * residual``.

The heads are identity-initialized (``strength ~ 0``, ``gain = 1``, zero residual), so an
untrained predictor returns ``z`` unchanged.

Public API:
  init_predictor_params(rng, *, latent_dim, action_dim, action_chunk_size, max_delay,
                        hidden_dim, num_streams=1, action_num_heads=4, action_num_layers=2)
  predict_latent_tokens(params, z[B,S,T,D], motion_actions[B,H,A], delay[B]) -> z_hat[B,S,T,D]
  predict_obs_latent(params, z_s[B,D], motion_actions[B,H,A], delay[B], *, max_delay) -> z_hat[B,D]
  save_predictor_checkpoint / load_predictor_checkpoint
"""
from __future__ import annotations

import pathlib
import pickle

import numpy as np


FORMAT_VERSION = 4
PREDICTOR_ARCHITECTURE = "motion_prior_singletoken_v1"

# Fixed knobs that survive at T=1 (single symbolic token).
_MAX_GAIN_DELTA = 1.0
_DEC_HIDDEN = 384
_TRUNK_BLOCKS = 2      # per-token residual FFN blocks over token_hidden
_DECODER_BLOCKS = 2    # per-token residual FFN blocks in the innovation decoder
_INTERACTION_GATE_BIAS = -4.0


# --- shared low-level helpers -------------------------------------------------


def _layer_norm(x, eps: float = 1e-6):
    import jax.numpy as jnp

    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(variance + eps)


def _token_grid_positions(token_count: int, *, dtype):
    import jax.numpy as jnp

    side = int(token_count**0.5)
    if side * side == token_count:
        axis = jnp.linspace(-1.0, 1.0, side, dtype=dtype)
        yy, xx = jnp.meshgrid(axis, axis, indexing="ij")
        return jnp.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    x = jnp.linspace(-1.0, 1.0, token_count, dtype=dtype)
    return jnp.stack([x, jnp.zeros_like(x)], axis=-1)


def _build_action_trajectory_features(motion_actions, delay):
    import jax.numpy as jnp

    step_ids = jnp.arange(motion_actions.shape[1])
    valid_mask = step_ids[None, :] < delay[:, None]
    actions = motion_actions * valid_mask[:, :, None].astype(motion_actions.dtype)
    previous = jnp.concatenate([jnp.zeros_like(actions[:, :1]), actions[:, :-1]], axis=1)
    action_change = (actions - previous) * valid_mask[:, :, None].astype(actions.dtype)
    absolute_residual = jnp.cumsum(actions, axis=1)
    path_magnitude = jnp.cumsum(jnp.abs(actions), axis=1)
    progress = (step_ids[None, :] + 1).astype(actions.dtype) / delay[:, None].astype(actions.dtype)
    progress = jnp.minimum(progress, 1.0) * valid_mask.astype(actions.dtype)
    features = jnp.concatenate(
        [
            actions,
            absolute_residual,
            action_change,
            path_magnitude,
            progress[:, :, None],
            valid_mask[:, :, None].astype(actions.dtype),
        ],
        axis=-1,
    )
    return features, valid_mask, absolute_residual


def _temporal_attention_block(x, valid_mask, block, num_heads: int):
    import jax
    import jax.numpy as jnp

    batch, steps, hidden = x.shape
    head_dim = hidden // num_heads
    norm_x = _layer_norm(x)
    qkv = norm_x @ block["qkv_w"] + block["qkv_b"]
    qkv = qkv.reshape(batch, steps, 3, num_heads, head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    logits = jnp.einsum("bthd,buhd->bhtu", q, k) / jnp.sqrt(head_dim)
    logits = jnp.where(valid_mask[:, None, None, :], logits, -1e9)
    weights = jax.nn.softmax(logits, axis=-1)
    attended = jnp.einsum("bhtu,buhd->bthd", weights, v).reshape(batch, steps, hidden)
    x = x + attended @ block["out_w"] + block["out_b"]
    ffn = jax.nn.silu(_layer_norm(x) @ block["ffn_w1"] + block["ffn_b1"])
    x = x + ffn @ block["ffn_w2"] + block["ffn_b2"]
    return x * valid_mask[:, :, None].astype(x.dtype)


def _residual_ffn_params(weight, width):
    """Params for one PerTokenResidualBlock: prenorm (parameterless) + FFN width->2w->width."""
    import jax.numpy as jnp

    return {
        "ffn_w1": weight((width, width * 2)),
        "ffn_b1": jnp.zeros((width * 2,)),
        "ffn_w2": weight((width * 2, width)),
        "ffn_b2": jnp.zeros((width,)),
    }


def _residual_ffn(x, block):
    """PerTokenResidualBlock forward: x + FFN(LN(x)). No token mixing."""
    import jax
    import jax.numpy as jnp

    h = jax.nn.silu(_layer_norm(x) @ block["ffn_w1"] + block["ffn_b1"])
    return x + h @ block["ffn_w2"] + block["ffn_b2"]


# --- model --------------------------------------------------------------------


def init_predictor_params(
    rng,
    *,
    latent_dim: int,
    action_dim: int,
    action_chunk_size: int,
    max_delay: int,
    hidden_dim: int,
    num_streams: int = 1,
    action_num_heads: int = 4,
    action_num_layers: int = 2,
):
    """Initialize the single-token predictor with identity-start heads."""
    import jax
    import jax.numpy as jnp

    if action_num_heads != 4:
        raise ValueError("predictor currently requires action_num_heads=4")
    if hidden_dim % action_num_heads:
        raise ValueError(f"hidden_dim={hidden_dim} must be divisible by {action_num_heads}")

    keys = iter(jax.random.split(rng, 128))

    def weight(shape, *, zero=False):
        if zero:
            return jnp.zeros(shape)
        return jax.random.normal(next(keys), shape) / jnp.sqrt(shape[0])

    action_blocks = []
    for _ in range(action_num_layers):
        action_blocks.append(
            {
                "qkv_w": weight((hidden_dim, hidden_dim * 3)),
                "qkv_b": jnp.zeros((hidden_dim * 3,)),
                "out_w": weight((hidden_dim, hidden_dim)),
                "out_b": jnp.zeros((hidden_dim,)),
                "ffn_w1": weight((hidden_dim, hidden_dim * 2)),
                "ffn_b1": jnp.zeros((hidden_dim * 2,)),
                "ffn_w2": weight((hidden_dim * 2, hidden_dim)),
                "ffn_b2": jnp.zeros((hidden_dim,)),
            }
        )

    action_feature_dim = action_dim * 4 + 2
    params = {
        # --- action trajectory encoder ---
        "delay_embedding": jax.random.normal(next(keys), (max_delay + 1, hidden_dim)) * 0.02,
        "stream_embedding": jax.random.normal(next(keys), (num_streams, hidden_dim)) * 0.02,
        "action_in_w1": weight((action_feature_dim, hidden_dim)),
        "action_in_b1": jnp.zeros((hidden_dim,)),
        "action_in_w2": weight((hidden_dim, hidden_dim)),
        "action_in_b2": jnp.zeros((hidden_dim,)),
        "action_blocks": action_blocks,
        "summary_w1": weight((hidden_dim * 2, hidden_dim)),
        "summary_b1": jnp.zeros((hidden_dim,)),
        "summary_w2": weight((hidden_dim, hidden_dim)),
        "summary_b2": jnp.zeros((hidden_dim,)),
        # --- context + token encoders ---
        "context_w": weight((latent_dim, hidden_dim)),
        "context_b": jnp.zeros((hidden_dim,)),
        "token_w": weight((latent_dim, hidden_dim)),
        "token_b": jnp.zeros((hidden_dim,)),
        "position_w1": weight((2, hidden_dim)),
        "position_b1": jnp.zeros((hidden_dim,)),
        "position_w2": weight((hidden_dim, hidden_dim)),
        "position_b2": jnp.zeros((hidden_dim,)),
        # --- per-token residual trunk (rr) ---
        "trunk_blocks": [_residual_ffn_params(weight, hidden_dim) for _ in range(_TRUNK_BLOCKS)],
        # --- transport strength (identity-closed) + gain head (gain=1 at init) ---
        "strength_w1": weight((hidden_dim, hidden_dim)),
        "strength_b1": jnp.zeros((hidden_dim,)),
        "strength_w2": weight((hidden_dim, 1), zero=True),
        "strength_b2": jnp.full((1,), -6.0),
        "gain_w1": weight((hidden_dim, hidden_dim)),
        "gain_b1": jnp.zeros((hidden_dim,)),
        "gain_w2": weight((hidden_dim, 1), zero=True),
        "gain_b2": jnp.zeros((1,)),  # tanh(0)=0 -> gain 1 at init
        # --- transport delta encoder ---
        "transport_delta_w": weight((latent_dim, hidden_dim)),
        "transport_delta_b": jnp.zeros((hidden_dim,)),
        # --- deeper rr residual (innovation) decoder @ dec_hidden ---
        "resin_w": weight((hidden_dim * 2, _DEC_HIDDEN)),
        "resin_b": jnp.zeros((_DEC_HIDDEN,)),
        "dec_blocks": [_residual_ffn_params(weight, _DEC_HIDDEN) for _ in range(_DECODER_BLOCKS)],
        "resout_w": weight((_DEC_HIDDEN, latent_dim + 1), zero=True),
        "resout_b": jnp.concatenate([jnp.zeros((latent_dim,)), jnp.full((1,), _INTERACTION_GATE_BIAS)]),
    }
    return params


def predict_latent_tokens(
    params,
    z,
    motion_actions,
    delay,
    *,
    return_aux: bool = False,
    max_gain_delta: float = _MAX_GAIN_DELTA,
    **_ignored,
):
    """Single-token forward. z: [B, S, T, D] (S=T=1 for Kinetix)."""
    import jax
    import jax.numpy as jnp

    if z.ndim != 4:
        raise ValueError(f"z must have shape [B, S, T, D], got {z.shape}")
    if motion_actions.ndim != 3:
        raise ValueError(f"motion_actions must have shape [B, H, A], got {motion_actions.shape}")
    if delay.ndim != 1 or delay.shape[0] != z.shape[0]:
        raise ValueError(f"delay must have shape [B], got {delay.shape}")
    if z.shape[1] > params["stream_embedding"].shape[0]:
        raise ValueError("z contains more streams than the predictor supports")

    # --- action trajectory encoder ---
    action_features, action_mask, absolute_residual = _build_action_trajectory_features(
        motion_actions, delay
    )
    seq = jax.nn.silu(action_features @ params["action_in_w1"] + params["action_in_b1"])
    seq = seq @ params["action_in_w2"] + params["action_in_b2"]
    seq = seq * action_mask[:, :, None].astype(seq.dtype)
    for block in params["action_blocks"]:
        seq = _temporal_attention_block(seq, action_mask, block, 4)
    idx = jnp.arange(z.shape[0])
    endpoint = seq[idx, delay - 1]
    pooled = seq.sum(axis=1) / delay[:, None].astype(seq.dtype)
    motion = jax.nn.silu(
        jnp.concatenate([pooled, endpoint], axis=-1) @ params["summary_w1"] + params["summary_b1"]
    )
    motion = motion @ params["summary_w2"] + params["summary_b2"]
    motion = motion + params["delay_embedding"][delay]

    # --- condition + token encoder ---
    stream_condition = params["stream_embedding"][: z.shape[1]][None, :, :]
    joint_context = jax.nn.silu(
        _layer_norm(z.mean(axis=(1, 2))) @ params["context_w"] + params["context_b"]
    )
    condition = motion[:, None, :] + stream_condition + joint_context[:, None, :]
    positions = _token_grid_positions(z.shape[2], dtype=z.dtype)
    pos_hidden = jax.nn.silu(positions @ params["position_w1"] + params["position_b1"])
    pos_hidden = pos_hidden @ params["position_w2"] + params["position_b2"]
    token_hidden = jax.nn.silu(_layer_norm(z) @ params["token_w"] + params["token_b"])
    token_hidden = token_hidden + condition[:, :, None, :] + pos_hidden[None, None]

    # --- per-token residual trunk (rr) ---
    for block in params["trunk_blocks"]:
        token_hidden = _residual_ffn(token_hidden, block)

    # --- transport: strength gate + per-token gain (warp is identity at T=1) ---
    strength_hidden = jax.nn.silu(token_hidden @ params["strength_w1"] + params["strength_b1"])
    strength = jax.nn.sigmoid(strength_hidden @ params["strength_w2"] + params["strength_b2"])
    gain_hidden = jax.nn.silu(token_hidden @ params["gain_w1"] + params["gain_b1"])
    gain = 1.0 + max_gain_delta * jnp.tanh(gain_hidden @ params["gain_w2"] + params["gain_b2"])
    z_transport = (1.0 - strength) * z + strength * (gain * z)

    # --- deeper rr residual (innovation) decoder + gate ---
    delta_hidden = jax.nn.silu(
        _layer_norm(z_transport - z) @ params["transport_delta_w"] + params["transport_delta_b"]
    )
    dec = jnp.concatenate([token_hidden, delta_hidden], axis=-1) @ params["resin_w"] + params["resin_b"]
    for block in params["dec_blocks"]:
        dec = _residual_ffn(dec, block)
    out = dec @ params["resout_w"] + params["resout_b"]
    interaction_residual = out[..., :-1]
    interaction_gate = jax.nn.sigmoid(out[..., -1:])
    z_hat = z_transport + interaction_gate * interaction_residual

    if not return_aux:
        return z_hat
    return z_hat, {
        "action_mask": action_mask,
        "absolute_residual_trajectory": absolute_residual,
        "token_positions": positions,
        "transport_strength": strength,
        "transport_gain": gain,
        "z_transport": z_transport,
        "interaction_gate": interaction_gate,
        "interaction_residual": interaction_residual,
    }


def predict_obs_latent(
    params,
    z_s,
    motion_actions,
    delay,
    *,
    max_delay: int,
    **kwargs,
):
    """Kinetix adapter: one symbolic latent vector as one stream and one token."""
    del max_delay
    if z_s.ndim != 2:
        raise ValueError(f"z_s must have shape [B, D], got {z_s.shape}")
    return predict_latent_tokens(
        params,
        z_s[:, None, None, :],
        motion_actions,
        delay,
        **kwargs,
    )[:, 0, 0, :]


# --- checkpoint I/O -----------------------------------------------------------


def save_predictor_checkpoint(path: str | pathlib.Path, *, params, metadata: dict) -> None:
    import jax

    payload = {
        "format_version": FORMAT_VERSION,
        "params": jax.tree.map(lambda x: np.asarray(x), params),
        "metadata": metadata,
    }
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_predictor_checkpoint(path: str | pathlib.Path):
    import jax
    import jax.numpy as jnp

    with pathlib.Path(path).open("rb") as f:
        payload = pickle.load(f)
    if payload.get("format_version") != FORMAT_VERSION:
        raise RuntimeError(
            f"unsupported predictor checkpoint format {payload.get('format_version')}"
        )
    return jax.tree.map(jnp.asarray, payload["params"]), payload["metadata"]
