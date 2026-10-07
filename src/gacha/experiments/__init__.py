"""Experiment registry. Each module exposes ``run(out_dir: Path) -> dict[str, float]``."""

from __future__ import annotations

import importlib
from pathlib import Path

from ._io import update_results_md

EXPERIMENTS: list[str] = [
    "e01_reproduce_paper",
    "e02_endfield_single",
    "e03_personal_state",
    "e04_carry_over",
    "e05_tail_risk",
    "e06_mc_convergence",
    "e07_sensitivity",
]


def run_experiment(name: str, out_dir: Path, results_md: Path | None = None) -> dict:
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}; known: {EXPERIMENTS}")
    module = importlib.import_module(f"gacha.experiments.{name}")
    headline = module.run(Path(out_dir))
    if results_md is not None:
        lines = [
            f"- {k}: {v:.10g}" if isinstance(v, float) else f"- {k}: {v}"
            for k, v in headline.items()
        ]
        update_results_md(Path(results_md), name, lines)
    return headline
