---
name: run-experiment
description: Use when running, adding, or changing an experiment script in this repo (src/gacharisk/experiments, results tables and figures, headline numbers in docs/results.md).
---

# Run or add an experiment

## Run

    uv run gacha-risk experiment <name> [--out results] [--results-md docs/results.md]
    uv run gacha-risk experiment all

Tables land in `results/tables/<name>_*.csv|.md`, figures in `results/figures/<name>_fig_*.png|.pdf`,
and the headline dict replaces the `## <name>` section of `docs/results.md`. `results/` is
git-ignored; `docs/results.md` is committed, so commit it after a rerun.

## Add

1. Create `src/gacharisk/experiments/eNN_short_name.py` exposing `run(out_dir: Path) -> dict[str, float]`.
2. Build models from `gacharisk.rules` / `gacharisk.models`, compute with `gacharisk.kernel`
   (`EnumeratedChain.from_model`, `hitting_time`, `state_values`, `success_within`), summarize
   with `gacharisk.risk.metrics`. No rule constants and no probability arithmetic of your own.
3. Call `apply_style()` once; write every table with `write_table(df, out_dir, "eNN_...")` and
   every figure with `save(fig, figures_dir, "eNN_fig_...")` (`ensure_dirs` gives the dirs).
4. Append the name to `EXPERIMENTS` in `src/gacharisk/experiments/__init__.py` (keep the order) and
   extend `test_registry_complete`.
5. Add a smoke test in `tests/test_experiments.py`: run into `tmp_path`, assert the expected
   table and figure files exist, and assert two or three headline inequalities.
6. Run it, open the PNGs, then commit code, test and `docs/results.md` together.

## Figure rules

- Colors from `gacharisk.plots.style.PALETTE` in slot order; at most four series per axis.
- One y axis per chart; heatmaps use the single-hue `Blues` map via `gacharisk.plots.heatmaps`.
- Legend whenever two or more series are drawn; axis labels carry units ("paid pulls").
- The title names what is plotted, not a conclusion.
