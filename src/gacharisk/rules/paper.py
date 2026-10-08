"""Representative schedule of Hou, Zhu & Zhang (Symmetry 2026), Eq. (1)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .schedule import ramp_schedule


@dataclass(frozen=True)
class PaperSchedule:
    base: float = 0.006
    soft_start: int = 73
    slope: float = 0.0585
    hard: int = 90

    @property
    def probs(self) -> tuple[float, ...]:
        return ramp_schedule(self.base, self.soft_start, self.slope, self.hard)

    def p(self, t: int) -> float:
        return self.probs[t]

    def hard_only(self) -> PaperSchedule:
        """Same base and hard pity, no soft-pity ramp (paper Table 1, row 2)."""
        return replace(self, slope=0.0)
