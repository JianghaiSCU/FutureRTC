import numpy as np
import pytest

from ours_pi05.models.corrector import Corrector


def test_identity_proxy_returns_last_committed_action():
    c = Corrector()
    base_state = np.arange(14, dtype=np.float32)
    committed = np.random.default_rng(0).normal(size=(10, 14)).astype(np.float32)
    out = c(base_state, committed, delay=10)
    np.testing.assert_allclose(out, committed[-1], atol=0)


def test_delay_zero_returns_base_state():
    """d=0 意味着没有已提交动作 —— state 就是当前 state。"""
    c = Corrector()
    base_state = np.arange(14, dtype=np.float32)
    out = c(base_state, np.zeros((0, 14), dtype=np.float32), delay=0)
    np.testing.assert_allclose(out, base_state, atol=0)


def test_uses_only_first_delay_actions():
    c = Corrector()
    base_state = np.zeros(14, dtype=np.float32)
    committed = np.zeros((10, 14), dtype=np.float32)
    committed[4] = 1.0   # d=5 时应取这一行
    committed[9] = 99.0  # 不该被取到
    out = c(base_state, committed, delay=5)
    np.testing.assert_allclose(out, np.ones(14, dtype=np.float32), atol=0)


def test_rejects_short_committed_buffer():
    c = Corrector()
    with pytest.raises(ValueError):
        c(np.zeros(14, np.float32), np.zeros((3, 14), np.float32), delay=10)


def test_rejects_negative_delay():
    """Negative delay must raise ValueError, not silently use negative indexing."""
    c = Corrector()
    base_state = np.arange(14, dtype=np.float32)
    committed = np.random.default_rng(42).normal(size=(10, 14)).astype(np.float32)
    with pytest.raises(ValueError, match="delay must be >= 0"):
        c(base_state, committed, delay=-1)


def test_rejects_transposed_committed_actions():
    """Transposed (14, d) array should be rejected, not silently reshaped."""
    c = Corrector()
    base_state = np.zeros(14, dtype=np.float32)
    # (14, 10) transposed array: would have 140 elements, reshape to (10, 14) if allowed
    transposed = np.ones((14, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="must be shape"):
        c(base_state, transposed, delay=5)


def test_residual_ckpt_not_implemented():
    """Residual MLP path should raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        Corrector(residual_ckpt="some_checkpoint.pt")


def test_output_mutation_does_not_affect_inputs():
    """Mutating the returned array must not affect input buffers."""
    c = Corrector()
    base_state_orig = np.arange(14, dtype=np.float32)
    base_state = base_state_orig.copy()
    committed_orig = np.random.default_rng(99).normal(size=(5, 14)).astype(np.float32)
    committed = committed_orig.copy()

    # Call corrector and mutate the result
    out = c(base_state, committed, delay=3)
    out[:] = 999.0

    # Verify inputs are unchanged
    np.testing.assert_allclose(base_state, base_state_orig, atol=0)
    np.testing.assert_allclose(committed, committed_orig, atol=0)
