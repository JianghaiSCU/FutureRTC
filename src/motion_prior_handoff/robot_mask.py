"""Per-level robot vs. environment mask over the Kinetix symbolic observation.

Robot dims = active, dynamic (inverse_mass > 0), role == 0 shapes that are not walls.
Observation layout: 9 non-wall polygons x 26 dims, then 4 circles x 18 dims (walls are
polygon indices {1, 2, 3}). This split is what lets FutureRTC take the robot
dims from forward-simulation and the environment dims from the learned predictor.

Ported from the vlash-kinetix ``compute_robot_indices.py`` observation split.
"""
import json

import jax.numpy as jnp


def compute_robot_mask(level_path: str, obs_dim: int = 679) -> jnp.ndarray:
    """Boolean (obs_dim,) array: True where the obs dim belongs to a robot body."""
    with open(level_path) as f:
        level_data = json.load(f)
    polygons = level_data["env_state"]["polygon"]
    circles = level_data["env_state"]["circle"]
    polygon_dim, circle_dim = 26, 18
    walls = {1, 2, 3}

    robot_polygons = [
        idx for idx, p in enumerate(polygons)
        if p.get("active", False) and p.get("role", -1) == 0
        and p.get("inverse_mass", 1) > 0 and idx not in walls
    ]
    robot_circles = [
        idx for idx, c in enumerate(circles)
        if c.get("active", False) and c.get("role", -1) == 0 and c.get("inverse_mass", 1) > 0
    ]
    polygon_to_obs_idx = {p: i for i, p in enumerate(j for j in range(12) if j not in walls)}

    mask = jnp.zeros(obs_dim, dtype=bool)
    for idx in robot_polygons:
        obs_idx = polygon_to_obs_idx[idx]
        mask = mask.at[obs_idx * polygon_dim:(obs_idx + 1) * polygon_dim].set(True)
    circle_start = 9 * polygon_dim
    for idx in robot_circles:
        mask = mask.at[circle_start + idx * circle_dim:circle_start + (idx + 1) * circle_dim].set(True)
    return mask
