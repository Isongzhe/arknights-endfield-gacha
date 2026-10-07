"""Endfield limited character banner as a finite DAG model (spec §5.5)."""

from __future__ import annotations

from collections.abc import Hashable
from typing import NamedTuple

from gacha.kernel.types import FAIL, SUCCESS, Transition
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules

ACTIVE = "active"
DONE = "success"
EXHAUSTED = "exhausted"


class BannerState(NamedTuple):
    t: int  # pity counter (carried across banners)
    n: int  # counted pulls in this banner, free and paid
    c: int  # UP copies obtained in this banner, capped at target_copies
    u: int  # 1 once a UP came from a counted pull (voids the 120 guarantee)


FRESH = BannerState(0, 0, 0, 0)


def pull(
    rules: EndfieldCharacterRules, state: BannerState, f: int, target_copies: int
) -> list[tuple[float, BannerState, int]]:
    """Distribution of the next state after one counted pull: (prob, next, cost)."""
    t, n, c, u = state
    cost = 0 if n < f else 1
    n1 = n + 1
    branches: list[tuple[float, int, int, int]] = []  # (prob, t', c', u')
    if rules.guarantee_pull is not None and u == 0 and n1 == rules.guarantee_pull:
        branches.append((1.0, 0, c + 1, 1))  # R5: forced UP, resets pity
    else:
        p = rules.p(t)
        q = rules.up_share
        if p * q > 0.0:
            branches.append((p * q, 0, c + 1, 1))  # R4: UP
        if p * (1.0 - q) > 0.0:
            branches.append((p * (1.0 - q), 0, c, u))  # off-rate 6*, R3 reset
        if p < 1.0:
            branches.append((1.0 - p, t + 1, c, u))
    potential = rules.potential_every is not None and n1 % rules.potential_every == 0
    bonus = 1 if potential else 0  # R8
    is_vacuum = n1 in rules.vacuum_points
    vacuum = rules.vacuum_copies_pmf() if is_vacuum else (1.0,)  # R6
    merged: dict[BannerState, float] = {}
    for pr, t1, c1, u1 in branches:
        for k, pk in enumerate(vacuum):
            if pk == 0.0:
                continue
            c2 = min(c1 + bonus + k, target_copies)
            key = BannerState(t1, n1, c2, u1)
            merged[key] = merged.get(key, 0.0) + pr * pk
    return [(pr, st, cost) for st, pr in merged.items()]


def status(rules: EndfieldCharacterRules, state: BannerState, spec: BannerSpec, f: int) -> str:
    """P3: success once the target is reached and free pulls are used up; exhausted at the cap."""
    _, n, c, _ = state
    reached = spec.target_copies >= 1 and c >= spec.target_copies
    if reached and n >= f:
        return DONE
    if not reached and n >= f and max(0, n - f) >= spec.cap:
        return EXHAUSTED  # free pulls are always used first (P1, P2), then the cap applies
    return ACTIVE


def validate_start(spec: BannerSpec, start: BannerState, f: int) -> None:
    t, n, c, u = start
    if not 0 <= t < spec.rules.hard_pity:
        raise ValueError(f"pity t must be in [0, {spec.rules.hard_pity}), got {t}")
    if n < 0:
        raise ValueError("banner pulls n must be >= 0")
    if not 0 <= c <= spec.target_copies:
        raise ValueError(f"copies c must be in [0, {spec.target_copies}], got {c}")
    if u not in (0, 1):
        raise ValueError("u must be 0 or 1")
    if max(0, n - f) > spec.cap:
        raise ValueError(f"start state has {max(0, n - f)} paid pulls, beyond the cap {spec.cap}")
    g = spec.rules.guarantee_pull
    if g is not None and u == 0 and n >= g:
        raise ValueError(
            f"n = {n} >= guarantee_pull = {g} requires u = 1: the {g}th counted pull always"
            " grants the UP (R5)"
        )


class SingleBannerModel:
    """One banner; absorbs success when the target is reached, fail when the cap is exhausted."""

    def __init__(self, spec: BannerSpec, start: BannerState = FRESH, dossier: bool = False):
        self.spec = spec
        self.rules = spec.rules
        self.f = spec.rules.free_pulls(dossier)
        self.start = BannerState(*start)
        validate_start(spec, self.start, self.f)

    def initial(self) -> Hashable:
        return self.start

    def step(self, s: Hashable) -> list[Transition]:
        st = status(self.rules, s, self.spec, self.f)
        if st == DONE:
            return [Transition(1.0, None, cost=0, absorb=SUCCESS)]
        if st == EXHAUSTED:
            return [Transition(1.0, None, cost=0, absorb=FAIL)]
        return [
            Transition(pr, nxt, cost=cost)
            for pr, nxt, cost in pull(self.rules, s, self.f, self.spec.target_copies)
        ]
