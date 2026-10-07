"""Cheapest top-up for a given number of pulls under a price menu.

Prices are data, not rules of the game: they vary by region and over time. The default menu is
the Taiwan (TWD) price list after first-purchase bonuses are used up (docs/rules/pricing.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

JADE_PER_PULL = 500
JADE_PER_STONE = 75


@dataclass(frozen=True)
class PriceMenu:
    """``stone_tiers``: (price, jade) offers that can be bought any number of times.
    ``packs``: (price, pulls) offers that can each be bought once."""

    stone_tiers: tuple[tuple[int, int], ...]
    packs: tuple[tuple[int, int], ...] = ()
    currency: str = "NT$"

    def __post_init__(self) -> None:
        if not self.stone_tiers:
            raise ValueError("a price menu needs at least one stone tier")
        if any(price <= 0 or jade <= 0 for price, jade in self.stone_tiers):
            raise ValueError("stone tiers need positive price and jade")
        if any(price <= 0 or pulls <= 0 for price, pulls in self.packs):
            raise ValueError("packs need positive price and pulls")

    @cached_property
    def _unit(self) -> int:
        from math import gcd

        unit = 0
        for _, jade in self.stone_tiers:
            unit = gcd(unit, jade)
        return unit

    def _jade_cost(self, jade: int) -> int:
        """Cheapest price for at least ``jade`` jade from the repeatable tiers."""
        if jade <= 0:
            return 0
        units = -(-jade // self._unit)
        tiers = [(price, j // self._unit) for price, j in self.stone_tiers]
        best = [0] * (units + 1)
        for u in range(1, units + 1):
            best[u] = min(price + best[max(0, u - step)] for price, step in tiers)
        return best[units]

    def min_cost(self, pulls: int, leftover_jade: int = 0, use_packs: bool = True) -> int:
        """Cheapest price for ``pulls`` more pulls, given jade already in hand that is not
        enough for a pull. Each pack may be used at most once."""
        if pulls <= 0:
            return 0
        packs = self.packs if use_packs else ()
        best: int | None = None
        for mask in range(1 << len(packs)):
            price = sum(packs[i][0] for i in range(len(packs)) if mask >> i & 1)
            got = sum(packs[i][1] for i in range(len(packs)) if mask >> i & 1)
            rest = max(0, pulls - got)
            total = price + self._jade_cost(rest * JADE_PER_PULL - leftover_jade if rest else 0)
            best = total if best is None else min(best, total)
        return int(best)

    def max_pulls(self, budget: int, leftover_jade: int = 0, use_packs: bool = True) -> int:
        """Most pulls that ``budget`` can buy."""
        k = 0
        while self.min_cost(k + 1, leftover_jade, use_packs) <= budget:
            k += 1
        return k


# Taiwan price list without first-purchase doubles: (TWD, jade) and one banner pack per limited
# banner (10 banner-bound pulls for NT$490).
ENDFIELD_TW_STANDARD = PriceMenu(
    stone_tiers=tuple(
        (price, stones * JADE_PER_STONE)
        for price, stones in ((2000, 242), (990, 112), (620, 68), (390, 40), (260, 26), (60, 6))
    ),
    packs=((490, 10),),
)
