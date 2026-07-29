"""Predictor unit tests (CPU-only; no RTC env or GPU needed)."""
import pathlib
import sys
import tempfile
import unittest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from motion_prior_handoff.predictor import (  # noqa: E402
    PREDICTOR_ARCHITECTURE,
    init_predictor_params,
    load_predictor_checkpoint,
    predict_latent_tokens,
    predict_obs_latent,
    save_predictor_checkpoint,
)

LATENT_DIM = 16
ACTION_DIM = 3
CHUNK = 8
MAX_DELAY = 4
HIDDEN = 32


def _params(seed=0):
    return init_predictor_params(
        jax.random.key(seed),
        latent_dim=LATENT_DIM, action_dim=ACTION_DIM, action_chunk_size=CHUNK,
        max_delay=MAX_DELAY, hidden_dim=HIDDEN,
    )


def _batch(batch_size=5, seed=1):
    keys = jax.random.split(jax.random.key(seed), 3)
    z_s = jax.random.normal(keys[0], (batch_size, LATENT_DIM))
    motion = jax.random.normal(keys[1], (batch_size, CHUNK, ACTION_DIM))
    delay = jnp.array([1, 2, 3, 4, 2][:batch_size], dtype=jnp.int32)
    # zero-pad actions beyond the delay, matching the collected shard format
    step_mask = jnp.arange(CHUNK)[None, :] < delay[:, None]
    motion = jnp.where(step_mask[:, :, None], motion, 0.0)
    return z_s, motion, delay


class PredictorTest(unittest.TestCase):
    def test_identity_initialization_returns_input(self):
        """Heads are identity-initialized, so an untrained predictor returns z unchanged."""
        params = _params()
        z_s, motion, delay = _batch()
        z_hat = predict_obs_latent(params, z_s, motion, delay, max_delay=MAX_DELAY)
        self.assertEqual(z_hat.shape, z_s.shape)
        self.assertTrue(jnp.allclose(z_hat, z_s, atol=1e-3),
                        msg=f"max abs diff {float(jnp.max(jnp.abs(z_hat - z_s)))}")

    def test_predict_latent_tokens_shape(self):
        params = _params()
        z_s, motion, delay = _batch()
        z = z_s[:, None, None, :]
        z_hat = predict_latent_tokens(params, z, motion, delay)
        self.assertEqual(z_hat.shape, z.shape)

    def test_return_aux_keys(self):
        params = _params()
        z_s, motion, delay = _batch()
        _, aux = predict_latent_tokens(params, z_s[:, None, None, :], motion, delay, return_aux=True)
        for key in ("transport_strength", "transport_gain", "interaction_gate", "interaction_residual"):
            self.assertIn(key, aux)

    def test_requires_four_heads(self):
        with self.assertRaises(ValueError):
            init_predictor_params(
                jax.random.key(0), latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
                action_chunk_size=CHUNK, max_delay=MAX_DELAY, hidden_dim=HIDDEN,
                action_num_heads=3,
            )

    def test_hidden_dim_divisible_by_heads(self):
        with self.assertRaises(ValueError):
            init_predictor_params(
                jax.random.key(0), latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
                action_chunk_size=CHUNK, max_delay=MAX_DELAY, hidden_dim=30,
            )

    def test_checkpoint_roundtrip(self):
        params = _params()
        metadata = {"level_name": "demo", "predictor_architecture": PREDICTOR_ARCHITECTURE,
                    "max_delay": MAX_DELAY, "obs_dim": 679}
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "demo.pkl"
            save_predictor_checkpoint(path, params=params, metadata=metadata)
            loaded_params, loaded_md = load_predictor_checkpoint(path)
        self.assertEqual(loaded_md, metadata)
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(loaded_params)):
            self.assertTrue(jnp.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
