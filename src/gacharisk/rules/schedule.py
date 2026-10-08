"""Piecewise-linear pity schedules shared by the paper and Endfield rules."""

from __future__ import annotations


def ramp_schedule(base: float, soft_start: int, step: float, hard: int) -> tuple[float, ...]:
    """Return p_t for t = 0..hard-1.

    p_t = base                                for t < soft_start
    p_t = base + step * (t - soft_start + 1)  for soft_start <= t < hard - 1
    p_{hard-1} = 1                            (hard pity)
    Values are clipped to [0, 1]. ``t`` is the number of consecutive failures so far.
    """
    if hard < 1:
        raise ValueError("hard must be >= 1")
    if not 0.0 < base <= 1.0:
        raise ValueError("base must be in (0, 1]")
    if step < 0.0:
        raise ValueError("step must be >= 0")
    probs: list[float] = []
    for t in range(hard - 1):
        p = base if t < soft_start else base + step * (t - soft_start + 1)
        probs.append(min(1.0, max(0.0, p)))
    probs.append(1.0)
    return tuple(probs)
