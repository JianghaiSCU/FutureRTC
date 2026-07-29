"""FlowPolicy observation-latent interface for RTC's symbolic Kinetix policy.

RTC's ``FlowPolicy`` has no standalone observation encoder: its first layer is linear over
``concat([noisy_action, obs])``. The observation contribution can therefore be separated
exactly, giving a policy-specific symbolic observation latent that is *linear* in the obs.
That linearity is what lets FutureRTC add a robot-from-sim latent and a
predicted-environment latent and decode a coherent action chunk:

    flow_obs_latent(obs) = in_proj([0, obs]) - in_proj([0, 0])
    z_robot + z_env      = flow_obs_latent(robot_obs) + flow_obs_latent(env_obs)

These functions import RTC's ``model`` module at call time, so ``bootstrap_rtc`` (see
``rtc_env``) must have run first. The level/delay helpers below are pure.
"""
from __future__ import annotations

import numpy as np


def level_name_from_path(level_path: str) -> str:
    return level_path.replace("/", "_").replace(".json", "")


def parse_level_indices(value: str, num_levels: int) -> list[int]:
    indices = [int(item) for item in value.split(",") if item]
    if not indices:
        raise ValueError("at least one level index is required")
    if len(set(indices)) != len(indices):
        raise ValueError(f"duplicate level indices are not allowed: {indices}")
    if min(indices) < 0 or max(indices) >= num_levels:
        raise ValueError(f"level indices must be within [0, {num_levels - 1}], got {indices}")
    return indices


def validate_delay(delay: int, action_chunk_size: int) -> None:
    if delay < 0:
        raise ValueError(f"delay must be non-negative, got {delay}")
    if delay > action_chunk_size:
        raise ValueError(
            f"delay={delay} requires {delay} executed actions, but action_chunk_size={action_chunk_size}"
        )


def pad_motion_actions_np(action_chunk: np.ndarray, delay: int) -> np.ndarray:
    """Keep the executed A1[s:s+d) actions and zero-pad the remaining chunk."""
    action_chunk = np.asarray(action_chunk)
    if action_chunk.ndim != 2:
        raise ValueError(f"action_chunk must have shape [H, A], got {action_chunk.shape}")
    validate_delay(delay, action_chunk.shape[0])
    output = np.zeros_like(action_chunk)
    output[:delay] = action_chunk[:delay]
    return output


def flow_obs_latent(policy, obs):
    """Extract the observation-only contribution of FlowPolicy.in_proj.

        in_proj([0, obs]) - in_proj([0, 0])

    Produces a policy-specific symbolic observation latent [B, channel_dim].
    """
    import jax.numpy as jnp

    if obs.ndim != 2:
        raise ValueError(f"obs must have shape [B, E], got {obs.shape}")
    zero_action = jnp.zeros((obs.shape[0], policy.action_dim), dtype=obs.dtype)
    zero_obs = jnp.zeros_like(obs)
    return policy.in_proj(jnp.concatenate([zero_action, obs], axis=-1)) - policy.in_proj(
        jnp.concatenate([zero_action, zero_obs], axis=-1)
    )


def flow_velocity_from_obs_latent(policy, obs_latent, x_t, time, obs_dim: int):
    """Run FlowPolicy using a predicted observation latent instead of a raw obs."""
    import functools

    import jax
    import jax.numpy as jnp

    import model as rtc_model

    if obs_latent.ndim != 2:
        raise ValueError(f"obs_latent must have shape [B, D], got {obs_latent.shape}")
    if x_t.shape != (obs_latent.shape[0], policy.action_chunk_size, policy.action_dim):
        raise ValueError(f"unexpected x_t shape {x_t.shape}")
    if time.ndim == 1:
        time = time[:, None]
    time = jnp.broadcast_to(time, (obs_latent.shape[0], policy.action_chunk_size))
    time_emb = jax.vmap(
        functools.partial(
            rtc_model.posemb_sincos,
            embedding_dim=policy.channel_dim,
            min_period=4e-3,
            max_period=4.0,
        )
    )(time)
    time_emb = policy.time_mlp(time_emb)

    zero_obs = jnp.zeros((obs_latent.shape[0], policy.action_chunk_size, obs_dim), dtype=x_t.dtype)
    x = policy.in_proj(jnp.concatenate([x_t, zero_obs], axis=-1))
    x = x + obs_latent[:, None, :]
    for mlp in policy.mlp_stack:
        x = mlp(x, time_emb)
    scale, shift = jnp.split(policy.final_adaln(time_emb), 2, axis=-1)
    x = policy.final_norm(x) * (1 + scale) + shift
    return policy.out_proj(x)


def action_from_obs_latent(policy, rng, obs_latent, num_steps: int, obs_dim: int, noise=None):
    """Sample an action chunk from FlowPolicy using a predicted observation latent."""
    import jax
    import jax.numpy as jnp

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if noise is None:
        noise = jax.random.normal(
            rng,
            shape=(obs_latent.shape[0], policy.action_chunk_size, policy.action_dim),
        )
    dt = 1.0 / num_steps

    def step(carry, _):
        x_t, time = carry
        v_t = flow_velocity_from_obs_latent(policy, obs_latent, x_t, time, obs_dim)
        return (x_t + dt * v_t, time + dt), None

    (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
    return x_1
