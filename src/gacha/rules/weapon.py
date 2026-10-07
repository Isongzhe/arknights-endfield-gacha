"""Arknights: Endfield weapon banner (Arsenal Exchange) parameters, rules W1-W6 in
docs/assumptions.md."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeaponBannerRules:
    six_rate: float = 0.04  # W2
    up_share: float = 0.25  # W2
    six_pity: int | None = 40  # W3: a 6* at the latest on the 40th pull since the last one
    up_guarantee: int = 80  # W4: the rate-up weapon at the latest on the 80th pull
    pulls_per_issue: int = 10  # W1: pulls only come in tens
    issue_cost: int = 1980  # W1: arsenal quota per ten-pull

    def __post_init__(self) -> None:
        if not 0.0 < self.six_rate <= 1.0 or not 0.0 < self.up_share <= 1.0:
            raise ValueError("six_rate and up_share must be in (0, 1]")
        if self.pulls_per_issue < 1 or self.issue_cost < 1:
            raise ValueError("pulls_per_issue and issue_cost must be >= 1")
        if self.up_guarantee % self.pulls_per_issue:
            raise ValueError("up_guarantee must be a whole number of issues")
        if self.six_pity is not None and self.six_pity < 1:
            raise ValueError("six_pity must be >= 1 or None")

    @property
    def max_issues(self) -> int:
        """Issues after which the rate-up weapon is certain."""
        return self.up_guarantee // self.pulls_per_issue
