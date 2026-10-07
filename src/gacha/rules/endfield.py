"""Arknights: Endfield limited character banner parameters (spec §3, rules R2-R9)."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

from .schedule import ramp_schedule


@dataclass(frozen=True)
class EndfieldCharacterRules:
    base_rate: float = 0.008  # R2
    soft_pity_start: int = 65  # R2: first t at which the ramp applies (66th pull)
    soft_pity_step: float = 0.05  # R2; set 0.0 for a flat schedule
    hard_pity: int = 80  # R2: p_{79} = 1
    up_share: float = 0.5  # R4
    guarantee_pull: int | None = 120  # R5
    vacuum_at: int | tuple[int, ...] | None = 30  # R6; a tuple gives several thresholds (RR3)
    vacuum_pulls: int = 10  # R6
    vacuum_rate: float = 0.008  # R6 (assumed flat)
    dossier_at: int | None = 60  # R7
    dossier_pulls: int = 10  # R7
    potential_every: int | None = 240  # R8
    free_start_pulls: int = 5  # R9
    five_star_rate: float = 0.08  # R11
    five_star_pity: int = 10  # R11: a 5* or better at least once every 10 pulls
    quota_per_permit: int = 25  # R13: 保障配額 for one universal permit
    quota_five_dupe: int = 10  # R13
    quota_six_dupe: int = 50  # R13

    def __post_init__(self) -> None:
        if not 0.0 <= self.up_share <= 1.0:
            raise ValueError("up_share must be in [0, 1]")
        if not 0.0 <= self.vacuum_rate <= 1.0:
            raise ValueError("vacuum_rate must be in [0, 1]")
        for name in ("free_start_pulls", "dossier_pulls", "vacuum_pulls"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("guarantee_pull", "dossier_at", "potential_every"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 or None")
        if any(v < 1 for v in self.vacuum_points):
            raise ValueError("vacuum_at thresholds must be >= 1")
        _ = self.probs  # validates base_rate, soft_pity_step, hard_pity

    @property
    def probs(self) -> tuple[float, ...]:
        return ramp_schedule(
            self.base_rate, self.soft_pity_start, self.soft_pity_step, self.hard_pity
        )

    def p(self, t: int) -> float:
        return self.probs[t]

    @property
    def vacuum_points(self) -> tuple[int, ...]:
        """Banner-local pull counts at which 10 uncounted bonus pulls are granted."""
        if self.vacuum_at is None:
            return ()
        if isinstance(self.vacuum_at, int):
            return (self.vacuum_at,)
        return tuple(self.vacuum_at)

    def free_pulls(self, dossier: bool) -> int:
        """Counted free pulls at the start of a banner (P1): 5 + 10 if a dossier is held."""
        return self.free_start_pulls + (self.dossier_pulls if dossier else 0)

    def vacuum_copies_pmf(self) -> tuple[float, ...]:
        """P(k UP copies from the 30-pull vacuum bonus), k = 0..vacuum_pulls (R6)."""
        if not self.vacuum_points:
            return (1.0,)
        r = self.vacuum_rate * self.up_share
        n = self.vacuum_pulls
        return tuple(comb(n, k) * r**k * (1.0 - r) ** (n - k) for k in range(n + 1))


def rerun_rules(**overrides) -> EndfieldCharacterRules:
    """Re-run banner (重構尋訪), rules RR1-RR7 in docs/assumptions.md.

    Counters persist across same-named re-runs: pass the saved state as the model's start.
    """
    params = {"vacuum_at": (30, 60, 90), "dossier_at": None, "free_start_pulls": 0}
    params.update(overrides)
    return EndfieldCharacterRules(**params)


@dataclass(frozen=True)
class BannerSpec:
    """One banner in a plan: rules, wanted copies (0 = skipped) and the paid-pull cap."""

    rules: EndfieldCharacterRules
    target_copies: int
    cap: int

    def __post_init__(self) -> None:
        if self.target_copies < 0:
            raise ValueError("target_copies must be >= 0")
        if self.cap < 0:
            raise ValueError("cap must be >= 0")
        if self.target_copies == 0 and self.cap != 0:
            raise ValueError("a skipped banner (target_copies=0) must have cap=0")
