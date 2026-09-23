"""Experiment registry. Each module exposes ``run(out_dir: Path) -> dict[str, float]``."""

from __future__ import annotations

import importlib
from pathlib import Path

from ._io import update_results_md

EXPERIMENTS: list[str] = []  # appended by Tasks 12-14


def run_experiment(name: str, out_dir: Path, results_md: Path | None = None) -> dict:
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}; known: {EXPERIMENTS}")
    module = importlib.import_module(f"gacha.experiments.{name}")
    headline = module.run(Path(out_dir))
    if results_md is not None:
        update_results_md(Path(results_md), name, [f"- {k}: {v}" for k, v in headline.items()])
    return headline
