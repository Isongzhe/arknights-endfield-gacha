"""Scenario files: a player's stock and banner progress as TOML (see examples/)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from gacharisk.cost.menu import JADE_PER_PULL, JADE_PER_STONE
from gacharisk.models.endfield import BannerState, SingleBannerModel
from gacharisk.rules.endfield import BannerSpec, EndfieldCharacterRules, rerun_rules

KINDS = {"limited": EndfieldCharacterRules, "rerun": rerun_rules}


@dataclass(frozen=True)
class BannerInput:
    kind: str
    rules: EndfieldCharacterRules
    start: BannerState

    def model(self) -> SingleBannerModel:
        """First-UP model that runs to the banner's guarantee."""
        cap = (self.rules.guarantee_pull or 0) + self.start.n + self.rules.hard_pity * 4
        return SingleBannerModel(BannerSpec(self.rules, 1, cap), start=self.start)


@dataclass(frozen=True)
class Scenario:
    stock_now: int
    future_pulls: int
    leftover_jade: int
    first: BannerInput
    second: BannerInput

    @property
    def stock_total(self) -> int:
        return self.stock_now + self.future_pulls


def _banner(table: dict, name: str) -> BannerInput:
    kind = table.get("kind", "limited")
    if kind not in KINDS:
        raise ValueError(f"[{name}] kind must be one of {sorted(KINDS)}, got {kind!r}")
    overrides = {}
    if "free_pulls" in table:
        overrides["free_start_pulls"] = int(table["free_pulls"])
    rules = KINDS[kind](**overrides)
    if "pulls_to_six_star" not in table:
        raise ValueError(f"[{name}] needs pulls_to_six_star (pulls left until the hard pity)")
    to_six = int(table["pulls_to_six_star"])
    if not 1 <= to_six <= rules.hard_pity:
        raise ValueError(f"[{name}] pulls_to_six_star must be in 1..{rules.hard_pity}")
    start = BannerState(rules.hard_pity - to_six, int(table.get("pulls_done", 0)), 0, 0)
    return BannerInput(kind, rules, start)


def load_scenario(path: Path) -> Scenario:
    data = tomllib.loads(Path(path).read_text())
    for section in ("stock", "first", "second"):
        if section not in data:
            raise ValueError(f"scenario file needs a [{section}] section")
    stock = data["stock"]
    jade = int(stock.get("jade", 0)) + int(stock.get("stones", 0)) * JADE_PER_STONE
    if jade < 0 or int(stock.get("tickets", 0)) < 0 or int(stock.get("future_pulls", 0)) < 0:
        raise ValueError("[stock] values must be >= 0")
    return Scenario(
        stock_now=int(stock.get("tickets", 0)) + jade // JADE_PER_PULL,
        future_pulls=int(stock.get("future_pulls", 0)),
        leftover_jade=jade % JADE_PER_PULL,
        first=_banner(data["first"], "first"),
        second=_banner(data["second"], "second"),
    )
