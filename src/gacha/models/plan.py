"""A chronological sequence of banners with carried pity and dossier (spec §5.6)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import NamedTuple

from gacha.kernel.types import FAIL, SUCCESS, Transition
from gacha.rules.endfield import BannerSpec

from .endfield import ACTIVE, DONE, BannerState, pull, status, validate_start


class PlanState(NamedTuple):
    k: int  # banner index
    t: int
    n: int
    c: int
    u: int
    d: int  # 1 if a dossier (10 free pulls) entered this banner


class Plan:
    """Player policy: pull each wanted banner until its target or its cap; skipped banners
    only use free pulls (P2). A wanted banner that exhausts its cap fails the whole plan."""

    def __init__(
        self,
        banners: Sequence[BannerSpec],
        start: BannerState = BannerState(0, 0, 0, 0),
        dossier0: bool = False,
    ):
        self.banners = list(banners)
        if not self.banners:
            raise ValueError("a plan needs at least one banner")
        self.start = BannerState(*start)
        self.dossier0 = 1 if dossier0 else 0
        first = self.banners[0]
        validate_start(first, self.start, first.rules.free_pulls(bool(self.dossier0)))

    def initial(self) -> Hashable:
        return PlanState(0, *self.start, self.dossier0)

    def step(self, s: Hashable) -> list[Transition]:
        k, t, n, c, u, d = s
        spec = self.banners[k]
        rules = spec.rules
        f = rules.free_pulls(bool(d))
        bs = BannerState(t, n, c, u)
        st = status(rules, bs, spec, f)
        if st == ACTIVE:
            return [
                Transition(pr, PlanState(k, *nxt, d), cost=cost)
                for pr, nxt, cost in pull(rules, bs, f, spec.target_copies)
            ]
        if st != DONE and spec.target_copies >= 1:
            return [Transition(1.0, None, cost=0, absorb=FAIL)]
        if k == len(self.banners) - 1:
            return [Transition(1.0, None, cost=0, absorb=SUCCESS)]
        d1 = 1 if (rules.dossier_at is not None and n >= rules.dossier_at) else 0  # R7
        return [Transition(1.0, PlanState(k + 1, t, 0, 0, 0, d1), cost=0)]
