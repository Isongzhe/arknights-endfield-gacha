"""Weapon banner as a finite DAG model. The unit of cost is one issue (a ten-pull)."""

from __future__ import annotations

from collections.abc import Hashable
from typing import NamedTuple

from gacha.kernel.types import FAIL, SUCCESS, Transition
from gacha.rules.weapon import WeaponBannerRules


class WeaponState(NamedTuple):
    n: int  # pulls made on this banner
    s: int  # pulls since the last 6* weapon


class WeaponBannerModel:
    """Pull until the rate-up weapon appears or ``cap_issues`` issues have been bought.

    The first pull of each issue carries cost 1, the other nine cost 0, so the hitting time is
    the number of issues paid for. An issue cannot be split, but the model stops as soon as the
    rate-up weapon appears because the rest of that issue no longer matters.
    """

    def __init__(
        self,
        rules: WeaponBannerRules,
        start: WeaponState = WeaponState(0, 0),  # noqa: B008
        cap_issues: int | None = None,
    ):
        self.rules = rules
        self.start = WeaponState(*start)
        k = rules.pulls_per_issue
        if self.start.n % k:
            raise ValueError("start must be on an issue boundary (a multiple of ten pulls)")
        if not 0 <= self.start.n < rules.up_guarantee:
            raise ValueError(f"pulls made must be in [0, {rules.up_guarantee})")
        if rules.six_pity is not None and not 0 <= self.start.s < rules.six_pity:
            raise ValueError(f"pulls since the last 6* must be in [0, {rules.six_pity})")
        left = (rules.up_guarantee - self.start.n) // k
        self.cap_issues = left if cap_issues is None else cap_issues
        if self.cap_issues < 0:
            raise ValueError("cap_issues must be >= 0")

    def initial(self) -> Hashable:
        return self.start

    def step(self, state: Hashable) -> list[Transition]:
        n, s = state
        r = self.rules
        boundary = n % r.pulls_per_issue == 0
        if boundary and (n - self.start.n) // r.pulls_per_issue >= self.cap_issues:
            return [Transition(1.0, None, cost=0, absorb=FAIL)]
        cost = 1 if boundary else 0
        n1 = n + 1
        if n1 == r.up_guarantee:
            return [Transition(1.0, None, cost=cost, absorb=SUCCESS)]
        p6 = 1.0 if (r.six_pity is not None and s == r.six_pity - 1) else r.six_rate
        out = [Transition(p6 * r.up_share, None, cost=cost, absorb=SUCCESS)]
        if p6 * (1.0 - r.up_share) > 0.0:
            out.append(Transition(p6 * (1.0 - r.up_share), WeaponState(n1, 0), cost=cost))
        if p6 < 1.0:
            out.append(Transition(1.0 - p6, WeaponState(n1, s + 1), cost=cost))
        return out
