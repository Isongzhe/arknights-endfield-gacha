"""Core kernel types: transitions, the model protocol and hitting-time results."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

SUCCESS = "success"
FAIL = "fail"


@dataclass(frozen=True)
class Transition:
    """One outgoing branch of a state.

    ``cost`` is the number of paid pulls consumed (0 or 1). Absorbing transitions have
    ``next=None`` and ``absorb`` set to SUCCESS or FAIL.
    """

    prob: float
    next: Hashable | None
    cost: int = 1
    absorb: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.prob <= 1.0 + 1e-12:
            raise ValueError("prob must be in [0, 1]")
        if self.cost not in (0, 1):
            raise ValueError("cost must be 0 or 1")
        if (self.next is None) != (self.absorb is not None):
            raise ValueError("absorbing transitions need next=None and an absorb label")
        if self.absorb not in (None, SUCCESS, FAIL):
            raise ValueError(f"unknown absorb label {self.absorb!r}")


class Model(Protocol):
    def initial(self) -> Hashable: ...

    def step(self, s: Hashable) -> list[Transition]: ...


@dataclass
class HittingTime:
    """Distribution of paid pulls until absorption.

    ``f_succ[j]`` / ``f_fail[j]`` = P(absorbed as success / fail with exactly j paid pulls).
    ``residual`` is the mass still unabsorbed at the horizon. ``residual_states`` maps each state
    reached after ``horizon`` paid pulls (and any free moves) to the probability that the path is
    there and its next move would be a paid pull.
    """

    f_succ: np.ndarray
    f_fail: np.ndarray
    residual: float
    residual_states: dict[Hashable, float] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return len(self.f_succ) - 1

    @property
    def pmf_stop(self) -> np.ndarray:
        return self.f_succ + self.f_fail

    @property
    def p_success(self) -> float:
        return float(self.f_succ.sum())

    @property
    def p_fail(self) -> float:
        return float(self.f_fail.sum())
