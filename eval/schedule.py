"""The delayed-handoff schedule for the predictor + corrector method.

With replan stride ``s`` and inference delay ``d``:

Window 0 is queried at obs(0) and executes chunk indices ``[0:s)`` over ``t`` in ``[0, s)``.
For each window ``m >= 1`` the query happens AT the handoff ``t = m*s``, but the observation
available at that moment is the one captured at ``t = m*s - d`` -- the delay is exactly what makes
it stale. The predictor forecasts the visual latent forward by ``d`` steps and the corrector
estimates the proprioceptive state at the handoff from the state at ``m*s - d`` plus the ``d``
actions committed in between. The policy is therefore queried with an ESTIMATE of the handoff
observation, and its fresh chunk executes from index 0 -- never a stale suffix.

Pure integer arithmetic (no policy, no simulator), so it is unit-testable in isolation.
"""
from __future__ import annotations


def _check(t: int, s: int) -> None:
    if t < 0 or s <= 0:
        raise ValueError(f"bad t/s: {t}/{s}")


def step_window(t: int, s: int) -> int:
    _check(t, s)
    return t // s


def step_index(t: int, s: int) -> int:
    """Chunk index executed at env step ``t``: every window executes its fresh chunk from 0."""
    _check(t, s)
    return t % s


def query_step(m: int, s: int) -> int:
    """Env time at which window ``m``'s chunk is queried."""
    if m < 0 or s <= 0:
        raise ValueError(f"bad args m={m} s={s}")
    return m * s


def stale_step(m: int, s: int, d: int) -> int:
    """Env time of the observation actually available at window ``m``'s query."""
    if m < 0 or s <= 0 or d < 0:
        raise ValueError(f"bad args m={m} s={s} d={d}")
    return m * s - d


def committed_slice(s: int, d: int) -> tuple[int, int]:
    """Slice of the previous window's chunk committed between the stale obs and the handoff."""
    if s <= 0 or d < 0:
        raise ValueError(f"bad args s={s} d={d}")
    if d > s:
        raise ValueError(f"delay {d} exceeds the replan stride {s}")
    return s - d, s


def compute_delay_batch_layout(num_delays: int, num_trials: int, nproc: int,
                               cpu_fraction: float = 0.6) -> int:
    """Trials per delay group, so every delay steps concurrently within a CPU budget.

    The eval runs all delays in ONE async vector-env batch of ``b_t * num_delays`` environments.
    ``b_t`` should divide ``num_trials`` so the trials partition evenly.
    """
    budget = int(cpu_fraction * nproc)
    b_t = min(num_trials, budget // num_delays)
    if b_t < 1:
        raise ValueError(f"CPU budget {budget} too small for {num_delays} delays")
    return b_t
