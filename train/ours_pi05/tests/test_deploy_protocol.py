"""Pins the HTTP protocol ``deploy_policy_server_ours_local.py`` (task 10) and
``deploy_policy_local_ours_local.py`` (task 11) must agree on.

Imports nothing hardware-dependent (no jax/openpi/piper_sdk/RealSense) -- only
``ours_pi05.deploy_protocol``, which itself only pulls in
``ours_pi05.action_space`` (numpy) and ``ours_pi05.dataset`` (numpy/torch, for
the ``MAX_DELAY`` constant). Runs anywhere, no GPU or robot required.

What actually breaks silently if the two deploy scripts drift apart (and what
each test below is standing in for):
  * client executes ``Cn[d:d+S]`` (the naive_async rule) instead of
    ``Cn[0:S]`` -> the method quietly degenerates into naive_async while still
    being reported as "ours".
  * client sends the wrong committed_actions slice -> the server's
    ``to_delta``/predictor conditioning is computed against actions that were
    never actually guaranteed to execute -> garbage z_hat, no crash.
  * server (or a copy-pasted version of this padding) right-aligns
    ``motion_actions`` instead of left-aligning -> predictor reads d rows of
    zeros as "valid" -> silently zero motion features, no crash (see
    ``ours_pi05/models/predictor.py`` valid_mask / endpoint indexing and
    ``ours_pi05/dataset.py`` PerFrameLatentDataset.__getitem__ comments).
"""

from __future__ import annotations

import numpy as np
import pytest

from ours_pi05.action_space import ACTION_DIM, qnorm, to_delta
from ours_pi05.dataset import MAX_DELAY
from ours_pi05.deploy_protocol import (
    CHUNK_LEN,
    DELAY_STEPS,
    FIRE_STEP,
    INTERP_STEP_SIZES,
    METHOD,
    S,
    build_padded_motion,
    committed_actions_from_slice,
    executed_slice,
    gap_log_filename,
    save_gap_log,
)


def _quantiles(rng):
    q01 = -rng.uniform(0.5, 2.0, size=ACTION_DIM).astype(np.float32)
    q99 = rng.uniform(0.5, 2.0, size=ACTION_DIM).astype(np.float32)
    return {"q01": q01, "q99": q99}


# ---------------------------------------------------------------------------
# Fixed timing contract (spec: S=25, d=10, H=50; client fires after S-d=15).
# ---------------------------------------------------------------------------


def test_timing_constants_match_spec():
    assert S == 25
    assert DELAY_STEPS == 10
    assert CHUNK_LEN == 50
    assert FIRE_STEP == S - DELAY_STEPS == 15


def test_method_identifier_is_ours():
    assert METHOD == "ours"


def test_interp_step_sizes_match_sync_and_rtc_clients_not_naive_async():
    """[0.015]*6 + [0.05] per arm, both arms -- NOT naive_async's flat 0.02."""
    assert INTERP_STEP_SIZES == (0.015,) * 6 + (0.05,) + (0.015,) * 6 + (0.05,)
    assert len(INTERP_STEP_SIZES) == ACTION_DIM
    assert 0.02 not in INTERP_STEP_SIZES


# ---------------------------------------------------------------------------
# Executed slice: the whole point of "ours" vs naive_async.
# ---------------------------------------------------------------------------


def test_executed_slice_is_0_to_S_not_d_to_dS():
    chunk = np.arange(CHUNK_LEN * ACTION_DIM, dtype=np.float32).reshape(CHUNK_LEN, ACTION_DIM)
    sliced = executed_slice(chunk)
    np.testing.assert_array_equal(sliced, chunk[0:S])
    # the naive_async rule would have been chunk[DELAY_STEPS:DELAY_STEPS + S];
    # assert this is a genuinely different slice, not an accidental match.
    naive_async_slice = chunk[DELAY_STEPS : DELAY_STEPS + S]
    assert not np.array_equal(sliced, naive_async_slice)
    assert sliced.shape == (S, ACTION_DIM)


def test_executed_slice_starts_at_row_zero():
    chunk = np.arange(CHUNK_LEN * ACTION_DIM, dtype=np.float32).reshape(CHUNK_LEN, ACTION_DIM)
    sliced = executed_slice(chunk)
    np.testing.assert_array_equal(sliced[0], chunk[0])  # not chunk[DELAY_STEPS]


def test_executed_slice_rejects_short_chunk():
    chunk = np.zeros((S - 1, ACTION_DIM), dtype=np.float32)
    with pytest.raises(ValueError):
        executed_slice(chunk)


# ---------------------------------------------------------------------------
# committed_actions = action_slice[15:25]
# ---------------------------------------------------------------------------


def test_committed_actions_is_slice_15_to_25():
    action_slice = np.arange(S * ACTION_DIM, dtype=np.float32).reshape(S, ACTION_DIM)
    committed = committed_actions_from_slice(action_slice)
    np.testing.assert_array_equal(committed, action_slice[15:25])
    assert committed.shape == (DELAY_STEPS, ACTION_DIM)


def test_committed_actions_last_row_is_final_raw_action_of_window():
    """a_{e-1}: the last row of committed_actions must be the very last
    (index S-1) raw action of the window -- what the gap log pairs against
    s_meas(e)."""
    action_slice = np.arange(S * ACTION_DIM, dtype=np.float32).reshape(S, ACTION_DIM)
    committed = committed_actions_from_slice(action_slice)
    np.testing.assert_array_equal(committed[-1], action_slice[S - 1])


def test_committed_actions_rejects_wrong_shape():
    short_slice = np.zeros((S - 1, ACTION_DIM), dtype=np.float32)
    with pytest.raises(ValueError):
        committed_actions_from_slice(short_slice)


# ---------------------------------------------------------------------------
# build_padded_motion: LEFT-aligned padding, exact match to
# PerFrameLatentDataset.__getitem__ / FastBatchLoader._fill_one.
# ---------------------------------------------------------------------------


def test_build_padded_motion_is_left_aligned():
    rng = np.random.default_rng(0)
    quantiles = _quantiles(rng)
    d = DELAY_STEPS
    committed = rng.normal(size=(d, ACTION_DIM)).astype(np.float32)
    s_anchor = rng.normal(size=ACTION_DIM).astype(np.float32)

    padded = build_padded_motion(committed, s_anchor, quantiles, d)

    assert padded.shape == (MAX_DELAY, ACTION_DIM)
    # valid region [0:d) must be non-trivially populated (not all-zero, since
    # committed is random)...
    assert not np.allclose(padded[:d], 0.0)
    # ...and must exactly equal qnorm(to_delta(committed, s_anchor)).
    expected = qnorm(to_delta(committed, s_anchor), quantiles)
    np.testing.assert_allclose(padded[:d], expected, atol=1e-6)
    # everything past d must be exactly zero (left-aligned padding).
    np.testing.assert_array_equal(padded[d:], np.zeros((MAX_DELAY - d, ACTION_DIM), dtype=np.float32))


def test_build_padded_motion_right_aligned_would_be_wrong():
    """Direct negative check for the documented failure mode: right-aligning
    hands the network d rows of zeros as its "valid" (first d) steps."""
    rng = np.random.default_rng(1)
    quantiles = _quantiles(rng)
    d = DELAY_STEPS
    committed = rng.normal(size=(d, ACTION_DIM)).astype(np.float32) + 5.0  # away from 0
    s_anchor = rng.normal(size=ACTION_DIM).astype(np.float32)

    padded = build_padded_motion(committed, s_anchor, quantiles, d)

    # A right-aligned implementation would have padded[:MAX_DELAY - d] == 0
    # and the real motion crammed into the tail -- assert our padding is NOT
    # that shape.
    wrong_right_aligned = np.zeros((MAX_DELAY, ACTION_DIM), dtype=np.float32)
    wrong_right_aligned[MAX_DELAY - d :] = qnorm(to_delta(committed, s_anchor), quantiles)
    assert not np.allclose(padded, wrong_right_aligned)
    # the actual first row (valid step 0) must be non-zero.
    assert not np.allclose(padded[0], 0.0)


def test_build_padded_motion_uses_stale_state_as_delta_anchor():
    """s_anchor for to_delta must be the STALE state, not some other
    quantity -- perturbing it must change the arm dims of the output (gripper
    dims 6/13 are untouched by to_delta, see action_space.GRIPPER_DIMS)."""
    rng = np.random.default_rng(2)
    quantiles = _quantiles(rng)
    d = DELAY_STEPS
    committed = rng.normal(size=(d, ACTION_DIM)).astype(np.float32)
    s_anchor = rng.normal(size=ACTION_DIM).astype(np.float32)

    base = build_padded_motion(committed, s_anchor, quantiles, d)
    perturbed = build_padded_motion(committed, s_anchor + 1.0, quantiles, d)
    assert not np.allclose(base[:d], perturbed[:d])


def test_build_padded_motion_rejects_delay_mismatch():
    rng = np.random.default_rng(3)
    quantiles = _quantiles(rng)
    committed = rng.normal(size=(DELAY_STEPS, ACTION_DIM)).astype(np.float32)
    s_anchor = rng.normal(size=ACTION_DIM).astype(np.float32)
    with pytest.raises(ValueError):
        build_padded_motion(committed, s_anchor, quantiles, DELAY_STEPS + 1)


def test_build_padded_motion_rejects_delay_out_of_range():
    rng = np.random.default_rng(4)
    quantiles = _quantiles(rng)
    s_anchor = rng.normal(size=ACTION_DIM).astype(np.float32)
    with pytest.raises(ValueError):
        build_padded_motion(np.zeros((0, ACTION_DIM), np.float32), s_anchor, quantiles, 0)
    with pytest.raises(ValueError):
        build_padded_motion(
            np.zeros((MAX_DELAY + 1, ACTION_DIM), np.float32), s_anchor, quantiles, MAX_DELAY + 1
        )


# ---------------------------------------------------------------------------
# End-to-end wiring: executed_slice -> committed_actions_from_slice ->
# build_padded_motion, exactly as the client fires it and the server consumes
# it, one full window.
# ---------------------------------------------------------------------------


def test_full_window_wiring_matches_client_and_server_usage():
    rng = np.random.default_rng(5)
    quantiles = _quantiles(rng)
    chunk = rng.normal(size=(CHUNK_LEN, ACTION_DIM)).astype(np.float32)
    s_anchor = rng.normal(size=ACTION_DIM).astype(np.float32)  # obs.state at fire time

    # client side
    action_slice = executed_slice(chunk)
    assert action_slice.shape == (S, ACTION_DIM)
    committed = committed_actions_from_slice(action_slice)

    # server side (payload's committed_actions is exactly `committed`)
    padded = build_padded_motion(committed, s_anchor, quantiles, DELAY_STEPS)
    assert padded.shape == (MAX_DELAY, ACTION_DIM)
    np.testing.assert_allclose(
        padded[:DELAY_STEPS], qnorm(to_delta(committed, s_anchor), quantiles), atol=1e-6
    )


# ---------------------------------------------------------------------------
# Gap log.
# ---------------------------------------------------------------------------


def test_gap_log_filename_format():
    assert gap_log_filename("plates_stacking", "20260714_120000") == (
        "gap_log_plates_stacking_20260714_120000.npz"
    )


def test_save_gap_log_round_trip(tmp_path):
    rng = np.random.default_rng(6)
    committed = rng.normal(size=(7, ACTION_DIM)).astype(np.float32)
    measured = rng.normal(size=(7, ACTION_DIM)).astype(np.float32)
    path = tmp_path / "gap_log_test.npz"

    save_gap_log(path, committed, measured)

    loaded = np.load(path)
    np.testing.assert_allclose(loaded["committed"], committed)
    np.testing.assert_allclose(loaded["measured"], measured)


def test_save_gap_log_rejects_shape_mismatch(tmp_path):
    committed = np.zeros((3, ACTION_DIM), dtype=np.float32)
    measured = np.zeros((4, ACTION_DIM), dtype=np.float32)
    with pytest.raises(ValueError):
        save_gap_log(tmp_path / "bad.npz", committed, measured)
