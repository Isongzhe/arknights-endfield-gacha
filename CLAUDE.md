# gacha-risk — exact gacha waiting-time and budget-risk models

Research codebase extending Hou, Zhu & Zhang (Symmetry 2026) to Arknights: Endfield. Goals: a
planning tool and a paper (`ROADMAP.md`, `docs/paper-outline.md`). Structure: `ARCHITECTURE.md`.
Rules and assumptions:
`docs/assumptions.md`. Research direction: `docs/research-notes.md`. Numbers: `docs/results.md`.

## Commands

    uv sync                      # environment
    uv run pytest                # all tests (paper reproduction + Monte Carlo parity included)
    uv run ruff check . && uv run ruff format .
    uv run gacha-risk evaluate --pity 40 --banner-pulls 20 --budget 60 --realized 90
    uv run gacha-risk experiment all  # tables -> results/tables, figures -> results/figures
    uv run gacha-risk evaluate --plan plan.toml --budget 300   # multi-banner; schema: gacha evaluate --help
    uv run gacha-risk decide examples/rerun_then_limited.toml  # two banners sharing one stock
    uv run python docs/site/build.py                      # rebuild the site (needs node for the engine check)

## Principles

- **One rule source, two engines.** Rules live in `src/gacharisk/rules/` (parameters) and
  `src/gacharisk/models/` (transitions). The exact engine (`kernel/`) and Monte Carlo (`mc/`) both
  consume `Model.step()`; never re-implement a rule in a simulator or an experiment.
- **T = paid pulls.** Free pulls are cost-0 transitions. Budgets, quantiles and VaR are in paid pulls.
- **Every model is a finite DAG.** Each transition must strictly increase a counter; the chain
  enumerator raises `CycleError` otherwise.
- **Assumptions are registered.** Any rule or policy change updates `docs/assumptions.md` first,
  then code, tests and affected experiments (use the `add-mechanic` skill).
- **Paper reproduction stays green.** `tests/test_paper_reproduction.py` pins the paper's numbers.
  Golden values (`tests/test_endfield_model.py::GOLDEN`) change only with a justification in the
  commit message and a note in `docs/research-notes.md`.
- **Experiments are reproducible scripts.** `src/gacharisk/experiments/eXX_name.py` exposes
  `run(out_dir) -> dict`; register it in `EXPERIMENTS`; write tables (CSV + MD) and figures (PNG +
  PDF); return headline numbers (use the `run-experiment` skill).
- **The site has a second engine.** `docs/site/engine.js` ports the first-UP recursion for the
  calculator; `docs/site/build.py` fails if it disagrees with Python. Change rules in Python
  first, then the port. Re-run rules and prices are in `docs/rules/`.
- **Figures** use `gacharisk.plots.style.PALETTE` in slot order, one axis per chart, a legend whenever
  two or more series are drawn, axis labels with units ("paid pulls").

## Layout

    src/gacharisk/rules      parameters (schedule.py, paper.py, endfield.py, weapon.py)
    src/gacharisk/kernel     types, chain enumeration, forward (PMF), backward (E, Var, success within b)
    src/gacharisk/models     paper.py, endfield.py (single banner), plan.py (multi-banner), weapon.py (issues)
    src/gacharisk/risk       metrics.py, normal.py
    src/gacharisk/mc         simulate.py, compare.py
    src/gacharisk/analysis   two_banner.py (stopping table for two banners sharing one stock), rebate.py (保障配額, arsenal income)
    src/gacharisk/cost       menu.py (price menus, cheapest top-up); prices are data, not rules
    src/gacharisk/scenario.py  TOML scenario files (examples/)
    src/gacharisk/plots      style.py, waiting_time.py, heatmaps.py
    src/gacharisk/experiments  registry + e01..e07
    tests/               one file per module; toy_models.py holds hand-checkable fixtures
