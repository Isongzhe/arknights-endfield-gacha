# Endfield Gacha Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 research codebase: an exact absorbing-chain engine, the Endfield limited character banner model with cross-banner carry-over, risk metrics, Monte Carlo parity, paper-reproduction tests, experiments with figures, a CLI, docs and two project skills.

**Architecture:** A generic finite absorbing-chain kernel (`gacha.kernel`) enumerates any model that exposes `initial()` and `step(state) -> list[Transition]`, computes the hitting-time distribution over *paid pulls* forward and expectations backward on the DAG. Rules live once (`gacha.rules`, `gacha.models`); Monte Carlo samples the same enumerated transitions. Risk metrics, plots, experiments and the CLI consume the kernel's `HittingTime`.

**Tech Stack:** Python 3.12, uv, numpy, scipy (sparse), pandas, matplotlib, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-16-endfield-gacha-phase1-design.md` (read it first; the plan argues from it).

## Global Constraints

- Python `>=3.12`; environment and lock managed by `uv` (`uv sync`, `uv run ...`). Dependencies: numpy, scipy, matplotlib, pandas; dev: pytest, ruff. No other runtime dependencies.
- `src/` layout, package name `gacha`. Experiments live in `src/gacha/experiments/` (spec §5.10 shows a top-level `experiments/`; it is moved inside the package so `gacha experiment` can import it without path hacks).
- Waiting time T counts **paid pulls only** (spec P4). Free pulls are transitions with `cost=0`.
- Game rules exist in exactly one place: `gacha/rules/*.py` parameters and `gacha/models/*.py` transitions. Monte Carlo never re-implements a rule.
- Every rule or policy change updates `docs/assumptions.md` (spec §3).
- Paper reproduction tolerances (spec §6): 0.01 on moments, exact on integer quantiles, 5e-4 on probabilities. A disagreement is investigated and documented, never forced.
- Monte Carlo parity: fixed seed, 200 000 paths, agreement within 3 standard errors.
- Models must be finite DAGs (every transition strictly increases a counter); the kernel raises `CycleError` otherwise.
- Figures use the validated categorical palette in `gacha/plots/style.py` in fixed slot order, one axis per chart, legend whenever two or more series are drawn, PNG and PDF written for every figure.
- Commit after every task with a conventional prefix (`feat:`, `test:`, `docs:`, `chore:`) and end each commit message with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. **Start state beyond the paid cap or with more copies than the target** — `SingleBannerModel(spec, start=...)` must raise `ValueError`, not silently produce a distribution. Pinned in Task 8 (`test_validate_start_rejects_bad_states`).
2. **Skipped banner with a positive cap** — `BannerSpec(rules, target_copies=0, cap=10)` must raise `ValueError`. Pinned in Task 2 (`test_banner_spec_validation`).
3. **Metrics on a truncated stop distribution** — a `HittingTime` computed with a horizon smaller than the support has `residual > 0`; `which="stop"` metrics must raise `ValueError` instead of reporting a biased mean, while `which="success"` still works. Pinned in Task 6 (`test_truncated_stop_distribution_rejected`).
4. **CLI called with copies already equal to the target, or a budget larger than the support** — must print `P(success) 1.0000` and completion 1, return 0, no traceback. Pinned in Task 11 (`test_cli_evaluate_already_owned`).
5. **Plan with zero banners, or whose last banner is skipped** — `Plan([])` raises `ValueError`; a plan ending in a skipped banner absorbs `success` after the free pulls. Pinned in Task 9 (`test_plan_rejects_empty`, `test_plan_last_skipped_banner_succeeds`).

## File map

| Path | Responsibility |
|------|----------------|
| `pyproject.toml`, `.gitignore` | uv project, scripts entry `gacha = "gacha.cli:main"` |
| `src/gacha/rules/schedule.py` | `ramp_schedule()` piecewise-linear pity schedule |
| `src/gacha/rules/paper.py` | `PaperSchedule` (Hou et al. representative schedule) |
| `src/gacha/rules/endfield.py` | `EndfieldCharacterRules`, `BannerSpec` |
| `src/gacha/kernel/types.py` | `Transition`, `Model`, `HittingTime`, `SUCCESS`, `FAIL` |
| `src/gacha/kernel/chain.py` | `EnumeratedChain`, `CycleError` |
| `src/gacha/kernel/forward.py` | `hitting_time()` |
| `src/gacha/kernel/backward.py` | `state_values()`, `success_within()` |
| `src/gacha/models/paper.py` | `SingleCounterModel`, `Featured5050Model`, `IIDStagesModel` |
| `src/gacha/models/endfield.py` | `BannerState`, `pull()`, `status()`, `validate_start()`, `SingleBannerModel` |
| `src/gacha/models/plan.py` | `PlanState`, `Plan` |
| `src/gacha/risk/metrics.py` | all distribution metrics, `convolve()`, `from_pmf()`, `summary()` |
| `src/gacha/risk/normal.py` | `normal_approx_cdf()`, `max_abs_cdf_error()` |
| `src/gacha/mc/simulate.py` | `Paths`, `simulate()` |
| `src/gacha/mc/compare.py` | `compare()` |
| `src/gacha/plots/style.py` | palette, `apply_style()`, `new_figure()`, `save()` |
| `src/gacha/plots/waiting_time.py` | `plot_pmf()`, `plot_cdf()`, `plot_survival()`, `plot_schedule()` |
| `src/gacha/plots/heatmaps.py` | `heatmap()` |
| `src/gacha/experiments/_io.py` | `ensure_dirs()`, `write_table()`, `update_results_md()` |
| `src/gacha/experiments/__init__.py` | `EXPERIMENTS`, `run_experiment()` |
| `src/gacha/experiments/e01..e07_*.py` | one `run(out_dir) -> dict` each |
| `src/gacha/cli.py` | `main(argv) -> int` |
| `docs/assumptions.md`, `docs/research-notes.md`, `docs/results.md`, `CLAUDE.md` | documentation |
| `.claude/skills/add-mechanic/SKILL.md`, `.claude/skills/run-experiment/SKILL.md` | project skills |

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/gacha/__init__.py`, `src/gacha/{rules,kernel,models,risk,mc,plots,experiments}/__init__.py`, `docs/results.md`
- Test: `tests/test_scaffold.py`

**Interfaces:**
- Produces: importable package `gacha` with `__version__ = "0.1.0"`; `uv run pytest` and `uv run gacha` work.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scaffold.py
def test_package_imports():
    import gacha

    assert gacha.__version__ == "0.1.0"
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "gacha"
version = "0.1.0"
description = "Exact waiting-time models for gacha banners (Arknights: Endfield) with cross-banner carry-over and risk metrics"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
  "numpy>=2.0",
  "scipy>=1.14",
  "matplotlib>=3.9",
  "pandas>=2.2",
]

[project.scripts]
gacha = "gacha.cli:main"

[dependency-groups]
dev = ["pytest>=8.3", "ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/gacha"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

- [ ] **Step 3: Create .gitignore, README.md, package skeleton and docs/results.md**

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
dist/
results/
```

`README.md`:
```markdown
# gacha

Exact waiting-time models for gacha banners, starting with the Arknights: Endfield limited
character banner. See `docs/superpowers/specs/` for the design and `CLAUDE.md` for conventions.

    uv sync
    uv run pytest
    uv run gacha evaluate --pity 40 --banner-pulls 20 --budget 60
    uv run gacha experiment all
```

`src/gacha/__init__.py`:
```python
"""Exact gacha waiting-time models."""

__version__ = "0.1.0"
```

Empty `__init__.py` (a one-line docstring) in `src/gacha/rules/`, `kernel/`, `models/`, `risk/`, `mc/`, `plots/`, `experiments/`. Also a placeholder `src/gacha/cli.py`:

```python
"""Command-line interface (filled in Task 11)."""


def main(argv: list[str] | None = None) -> int:
    print("gacha CLI not implemented yet")
    return 0
```

`docs/results.md`:
```markdown
# Results

Headline numbers written by `uv run gacha experiment <name>`. Each `##` section is owned by one
experiment and is replaced when it reruns.
```

- [ ] **Step 4: Install and run the test**

Run: `uv sync && uv run pytest tests/test_scaffold.py -v`
Expected: PASS (uv creates `.venv` and `uv.lock`).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .gitignore README.md src tests docs/results.md
git commit -m "chore: scaffold uv project and package skeleton

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Rule parameters (schedule, paper, Endfield)

**Files:**
- Create: `src/gacha/rules/schedule.py`, `src/gacha/rules/paper.py`, `src/gacha/rules/endfield.py`
- Test: `tests/test_rules.py`

**Interfaces:**
- Produces: `ramp_schedule(base, soft_start, step, hard) -> tuple[float, ...]`; `PaperSchedule(base=0.006, soft_start=73, slope=0.0585, hard=90)` with `.probs`, `.p(t)`, `.hard_only()`; `EndfieldCharacterRules(...)` with `.probs`, `.p(t)`, `.free_pulls(dossier: bool) -> int`, `.vacuum_copies_pmf() -> tuple[float, ...]`; `BannerSpec(rules, target_copies, cap)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rules.py
import math

import pytest

from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules
from gacha.rules.paper import PaperSchedule
from gacha.rules.schedule import ramp_schedule


def test_ramp_schedule_paper_values():
    p = ramp_schedule(0.006, 73, 0.0585, 90)
    assert len(p) == 90
    assert p[72] == pytest.approx(0.006)
    assert p[73] == pytest.approx(0.0645)
    assert p[88] == pytest.approx(0.006 + 16 * 0.0585)
    assert p[89] == 1.0


def test_ramp_schedule_no_ramp_when_step_zero():
    p = ramp_schedule(0.006, 73, 0.0, 90)
    assert all(x == pytest.approx(0.006) for x in p[:89])
    assert p[89] == 1.0


def test_ramp_schedule_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ramp_schedule(0.0, 1, 0.0, 10)
    with pytest.raises(ValueError):
        ramp_schedule(0.5, 1, 0.0, 0)
    with pytest.raises(ValueError):
        ramp_schedule(0.5, 1, -0.1, 10)


def test_paper_schedule_defaults_and_hard_only():
    s = PaperSchedule()
    assert s.p(0) == pytest.approx(0.006)
    assert s.p(89) == 1.0
    h = s.hard_only()
    assert h.p(88) == pytest.approx(0.006)
    assert h.p(89) == 1.0


def test_endfield_schedule_key_values():
    r = EndfieldCharacterRules()
    assert len(r.probs) == 80
    assert r.p(64) == pytest.approx(0.008)
    assert r.p(65) == pytest.approx(0.058)
    assert r.p(78) == pytest.approx(0.708)
    assert r.p(79) == 1.0


def test_endfield_free_pulls():
    r = EndfieldCharacterRules()
    assert r.free_pulls(False) == 5
    assert r.free_pulls(True) == 15
    assert EndfieldCharacterRules(free_start_pulls=0).free_pulls(True) == 10


def test_vacuum_copies_pmf_is_binomial():
    r = EndfieldCharacterRules()
    pmf = r.vacuum_copies_pmf()
    assert len(pmf) == 11
    assert sum(pmf) == pytest.approx(1.0)
    assert pmf[0] == pytest.approx((1 - 0.008 * 0.5) ** 10)
    assert pmf[1] == pytest.approx(10 * 0.004 * 0.996**9)
    assert EndfieldCharacterRules(vacuum_at=None).vacuum_copies_pmf() == (1.0,)


def test_endfield_rules_validation():
    with pytest.raises(ValueError):
        EndfieldCharacterRules(up_share=1.5)
    with pytest.raises(ValueError):
        EndfieldCharacterRules(guarantee_pull=0)
    with pytest.raises(ValueError):
        EndfieldCharacterRules(free_start_pulls=-1)


def test_banner_spec_validation():
    r = EndfieldCharacterRules()
    BannerSpec(r, target_copies=1, cap=120)
    BannerSpec(r, target_copies=0, cap=0)
    with pytest.raises(ValueError):
        BannerSpec(r, target_copies=0, cap=10)
    with pytest.raises(ValueError):
        BannerSpec(r, target_copies=-1, cap=0)
    with pytest.raises(ValueError):
        BannerSpec(r, target_copies=1, cap=-5)
    assert math.isfinite(r.p(0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rules.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gacha.rules.endfield'`.

- [ ] **Step 3: Implement the rule modules**

`src/gacha/rules/schedule.py`:
```python
"""Piecewise-linear pity schedules shared by the paper and Endfield rules."""

from __future__ import annotations


def ramp_schedule(base: float, soft_start: int, step: float, hard: int) -> tuple[float, ...]:
    """Return p_t for t = 0..hard-1.

    p_t = base                                for t < soft_start
    p_t = base + step * (t - soft_start + 1)  for soft_start <= t < hard - 1
    p_{hard-1} = 1                            (hard pity)
    Values are clipped to [0, 1]. ``t`` is the number of consecutive failures so far.
    """
    if hard < 1:
        raise ValueError("hard must be >= 1")
    if not 0.0 < base <= 1.0:
        raise ValueError("base must be in (0, 1]")
    if step < 0.0:
        raise ValueError("step must be >= 0")
    probs: list[float] = []
    for t in range(hard - 1):
        p = base if t < soft_start else base + step * (t - soft_start + 1)
        probs.append(min(1.0, max(0.0, p)))
    probs.append(1.0)
    return tuple(probs)
```

`src/gacha/rules/paper.py`:
```python
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
```

`src/gacha/rules/endfield.py`:
```python
"""Arknights: Endfield limited character banner parameters (spec §3, rules R2-R9)."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

from .schedule import ramp_schedule


@dataclass(frozen=True)
class EndfieldCharacterRules:
    base_rate: float = 0.008  # R2
    soft_pity_start: int = 65  # R2: first t at which the ramp applies (66th pull)
    soft_pity_step: float = 0.05  # R2
    hard_pity: int = 80  # R2: p_{79} = 1
    up_share: float = 0.5  # R4
    guarantee_pull: int | None = 120  # R5
    vacuum_at: int | None = 30  # R6
    vacuum_pulls: int = 10  # R6
    vacuum_rate: float = 0.008  # R6 (assumed flat)
    dossier_at: int | None = 60  # R7
    dossier_pulls: int = 10  # R7
    potential_every: int | None = 240  # R8
    free_start_pulls: int = 5  # R9

    def __post_init__(self) -> None:
        if not 0.0 <= self.up_share <= 1.0:
            raise ValueError("up_share must be in [0, 1]")
        if not 0.0 <= self.vacuum_rate <= 1.0:
            raise ValueError("vacuum_rate must be in [0, 1]")
        for name in ("free_start_pulls", "dossier_pulls", "vacuum_pulls"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("guarantee_pull", "vacuum_at", "dossier_at", "potential_every"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 or None")
        _ = self.probs  # validates base_rate, soft_pity_step, hard_pity

    @property
    def probs(self) -> tuple[float, ...]:
        return ramp_schedule(
            self.base_rate, self.soft_pity_start, self.soft_pity_step, self.hard_pity
        )

    def p(self, t: int) -> float:
        return self.probs[t]

    def free_pulls(self, dossier: bool) -> int:
        """Counted free pulls at the start of a banner (P1): 5 + 10 if a dossier is held."""
        return self.free_start_pulls + (self.dossier_pulls if dossier else 0)

    def vacuum_copies_pmf(self) -> tuple[float, ...]:
        """P(k UP copies from the 30-pull vacuum bonus), k = 0..vacuum_pulls (R6)."""
        if self.vacuum_at is None:
            return (1.0,)
        r = self.vacuum_rate * self.up_share
        n = self.vacuum_pulls
        return tuple(comb(n, k) * r**k * (1.0 - r) ** (n - k) for k in range(n + 1))


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rules.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gacha/rules tests/test_rules.py
git commit -m "feat(rules): pity schedules, paper schedule and Endfield banner parameters

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Kernel types and chain enumeration

**Files:**
- Create: `src/gacha/kernel/types.py`, `src/gacha/kernel/chain.py`
- Test: `tests/test_chain.py`, `tests/toy_models.py` (shared fixtures)

**Interfaces:**
- Produces: `SUCCESS = "success"`, `FAIL = "fail"`; `Transition(prob, next, cost=1, absorb=None)`; `Model` protocol (`initial()`, `step(s)`); `HittingTime(f_succ, f_fail, residual, residual_states)` with `.horizon`, `.pmf_stop`, `.p_success`, `.p_fail`; `CycleError`; `EnumeratedChain.from_model(model, roots=None)` with attributes `states, index, roots, n, tr_ptr, tr_prob, tr_next, tr_cost, tr_absorb, Q0T, Q1T, a0_succ, a0_fail, a1_succ, a1_fail, paid_out, topo, max_cost_path` and method `point_mass(state) -> np.ndarray`.
- Absorb codes in `tr_absorb`: 0 none, 1 success, 2 fail. `tr_next == -1` for absorbing transitions.

- [ ] **Step 1: Write the shared toy models and the failing tests**

```python
# tests/toy_models.py
"""Tiny hand-checkable models used by the kernel tests."""

from gacha.kernel.types import FAIL, SUCCESS, Transition


class ToyModel:
    """a -(free)-> b; b: success paid 0.6 / c paid 0.4; c: fail free 0.5 / success paid 0.5.

    Hand results: f_succ = [0, 0.6, 0.2], f_fail = [0, 0.2, 0]; P(success) = 0.8;
    E[T_stop] = 1.2, Var = 0.16; E[b] = 1.2, E[c] = 0.5, Var[c] = 0.25; max_cost_path = 2.
    """

    def initial(self):
        return "a"

    def step(self, s):
        if s == "a":
            return [Transition(1.0, "b", cost=0)]
        if s == "b":
            return [Transition(0.6, None, cost=1, absorb=SUCCESS), Transition(0.4, "c", cost=1)]
        if s == "c":
            return [Transition(0.5, None, cost=0, absorb=FAIL), Transition(0.5, None, cost=1, absorb=SUCCESS)]
        raise KeyError(s)


class CyclicModel:
    def initial(self):
        return "x"

    def step(self, s):
        if s == "x":
            return [Transition(0.5, "y", cost=1), Transition(0.5, None, cost=1, absorb=SUCCESS)]
        return [Transition(1.0, "x", cost=1)]


class BadProbModel:
    def initial(self):
        return 0

    def step(self, s):
        return [Transition(0.7, None, cost=1, absorb=SUCCESS)]


class DuplicateTargetModel:
    """Two transitions to the same (next, cost, absorb) must be merged by the chain."""

    def initial(self):
        return 0

    def step(self, s):
        if s == 0:
            return [Transition(0.25, 1, cost=1), Transition(0.25, 1, cost=1), Transition(0.5, None, cost=1, absorb=SUCCESS)]
        return [Transition(1.0, None, cost=1, absorb=SUCCESS)]
```

```python
# tests/test_chain.py
import numpy as np
import pytest

from gacha.kernel.chain import CycleError, EnumeratedChain
from gacha.kernel.types import FAIL, SUCCESS, HittingTime, Transition
from tests.toy_models import BadProbModel, CyclicModel, DuplicateTargetModel, ToyModel


def test_transition_validation():
    with pytest.raises(ValueError):
        Transition(0.5, None, cost=1)  # absorbing needs a label
    with pytest.raises(ValueError):
        Transition(0.5, "s", cost=1, absorb=SUCCESS)  # non-absorbing must not carry a label
    with pytest.raises(ValueError):
        Transition(0.5, "s", cost=2)
    with pytest.raises(ValueError):
        Transition(-0.1, "s")
    with pytest.raises(ValueError):
        Transition(0.5, None, absorb="other")


def test_hitting_time_properties():
    ht = HittingTime(np.array([0.0, 0.6, 0.2]), np.array([0.0, 0.2, 0.0]), 0.0)
    assert ht.horizon == 2
    assert ht.p_success == pytest.approx(0.8)
    assert ht.p_fail == pytest.approx(0.2)
    assert ht.pmf_stop.tolist() == pytest.approx([0.0, 0.8, 0.2])


def test_enumerates_toy_model():
    chain = EnumeratedChain.from_model(ToyModel())
    assert chain.n == 3
    assert chain.states[chain.roots[0]] == "a"
    assert set(chain.states) == {"a", "b", "c"}
    assert chain.max_cost_path == 2
    # a comes before b before c in topological order
    pos = {chain.states[i]: k for k, i in enumerate(chain.topo.tolist())}
    assert pos["a"] < pos["b"] < pos["c"]


def test_transition_arrays_and_vectors():
    chain = EnumeratedChain.from_model(ToyModel())
    b, c = chain.index["b"], chain.index["c"]
    assert chain.a1_succ[b] == pytest.approx(0.6)
    assert chain.a0_fail[c] == pytest.approx(0.5)
    assert chain.a1_succ[c] == pytest.approx(0.5)
    assert chain.paid_out[b] == pytest.approx(1.0)
    assert chain.paid_out[c] == pytest.approx(0.5)
    assert chain.paid_out[chain.index["a"]] == pytest.approx(0.0)
    # Q0T[b, a] = 1 (a -> b free); Q1T[c, b] = 0.4 (b -> c paid)
    assert chain.Q0T[b, chain.index["a"]] == pytest.approx(1.0)
    assert chain.Q1T[c, b] == pytest.approx(0.4)
    v = chain.point_mass("b")
    assert v.sum() == 1.0 and v[b] == 1.0


def test_merges_duplicate_targets():
    chain = EnumeratedChain.from_model(DuplicateTargetModel())
    root = chain.roots[0]
    k = slice(chain.tr_ptr[root], chain.tr_ptr[root + 1])
    assert chain.tr_prob[k].size == 2
    assert sorted(chain.tr_prob[k].tolist()) == pytest.approx([0.5, 0.5])


def test_cycle_detection():
    with pytest.raises(CycleError):
        EnumeratedChain.from_model(CyclicModel())


def test_probability_validation():
    with pytest.raises(ValueError):
        EnumeratedChain.from_model(BadProbModel())


def test_multiple_roots():
    chain = EnumeratedChain.from_model(ToyModel(), roots=["b", "c"])
    assert set(chain.states) == {"b", "c"}
    assert len(chain.roots) == 2
    assert chain.max_cost_path == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gacha.kernel.types'`.

- [ ] **Step 3: Implement types.py**

```python
# src/gacha/kernel/types.py
"""Core kernel types: transitions, the model protocol and hitting-time results."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

SUCCESS = "success"
FAIL = "fail"


@dataclass(frozen=True)
class Transition:
    """One outgoing branch of a state.

    ``cost`` is the number of paid pulls consumed (0 or 1). Absorbing transitions have
    ``next=None`` and ``absorb`` set to SUCCESS or FAIL.
    """

    prob: float
    next: Hashable | None
    cost: int = 1
    absorb: str | None = None

    def __post_init__(self) -> None:
        if self.prob < 0.0:
            raise ValueError("prob must be >= 0")
        if self.cost not in (0, 1):
            raise ValueError("cost must be 0 or 1")
        if (self.next is None) != (self.absorb is not None):
            raise ValueError("absorbing transitions need next=None and an absorb label")
        if self.absorb not in (None, SUCCESS, FAIL):
            raise ValueError(f"unknown absorb label {self.absorb!r}")


class Model(Protocol):
    def initial(self) -> Hashable: ...

    def step(self, s: Hashable) -> list[Transition]: ...


@dataclass
class HittingTime:
    """Distribution of paid pulls until absorption.

    ``f_succ[j]`` / ``f_fail[j]`` = P(absorbed as success / fail with exactly j paid pulls).
    ``residual`` is the mass still unabsorbed at the horizon; ``residual_states`` maps the
    states holding that mass to their probability.
    """

    f_succ: np.ndarray
    f_fail: np.ndarray
    residual: float
    residual_states: dict[Hashable, float] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return len(self.f_succ) - 1

    @property
    def pmf_stop(self) -> np.ndarray:
        return self.f_succ + self.f_fail

    @property
    def p_success(self) -> float:
        return float(self.f_succ.sum())

    @property
    def p_fail(self) -> float:
        return float(self.f_fail.sum())
```

- [ ] **Step 4: Implement chain.py**

```python
# src/gacha/kernel/chain.py
"""Enumerate a model's reachable states into arrays and sparse matrices."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .types import FAIL, SUCCESS, Model

ABSORB_CODE = {None: 0, SUCCESS: 1, FAIL: 2}
PROB_TOL = 1e-9


class CycleError(ValueError):
    """The state graph has a cycle; models must strictly advance a counter on every step."""


@dataclass
class EnumeratedChain:
    states: list[Hashable]
    index: dict[Hashable, int]
    roots: list[int]
    tr_ptr: np.ndarray  # (n+1,) CSR row pointers into the transition arrays
    tr_prob: np.ndarray  # (m,) float
    tr_next: np.ndarray  # (m,) int64, -1 when absorbing
    tr_cost: np.ndarray  # (m,) int8, 0 or 1
    tr_absorb: np.ndarray  # (m,) int8, 0 none / 1 success / 2 fail
    Q0T: sparse.csr_matrix  # transpose of the cost-0 state->state matrix
    Q1T: sparse.csr_matrix  # transpose of the cost-1 state->state matrix
    a0_succ: np.ndarray
    a0_fail: np.ndarray
    a1_succ: np.ndarray
    a1_fail: np.ndarray
    paid_out: np.ndarray  # (n,) total probability of cost-1 transitions out of each state
    topo: np.ndarray  # topological order (sources first)
    max_cost_path: int  # longest cost-weighted path from any root = support bound of T

    @property
    def n(self) -> int:
        return len(self.states)

    def point_mass(self, state: Hashable) -> np.ndarray:
        v = np.zeros(self.n)
        v[self.index[state]] = 1.0
        return v

    @classmethod
    def from_model(cls, model: Model, roots: Sequence[Hashable] | None = None) -> EnumeratedChain:
        root_states = [model.initial()] if roots is None else list(roots)
        if not root_states:
            raise ValueError("at least one root state is required")
        states: list[Hashable] = []
        index: dict[Hashable, int] = {}
        queue: deque[Hashable] = deque()
        for s in root_states:
            if s not in index:
                index[s] = len(states)
                states.append(s)
                queue.append(s)
        root_idx = [index[s] for s in root_states]

        # BFS: states are appended at discovery and popped FIFO, so rows[i] belongs to states[i].
        rows: list[list[tuple[float, int, int, int]]] = []
        while queue:
            s = queue.popleft()
            trs = model.step(s)
            total = sum(tr.prob for tr in trs)
            if abs(total - 1.0) > PROB_TOL:
                raise ValueError(f"transition probabilities from {s!r} sum to {total!r}, not 1")
            merged: dict[tuple[Hashable | None, int, str | None], float] = {}
            for tr in trs:
                if tr.prob == 0.0:
                    continue
                key = (tr.next, tr.cost, tr.absorb)
                merged[key] = merged.get(key, 0.0) + tr.prob
            row = []
            for (nxt, cost, absorb), p in merged.items():
                if nxt is None:
                    j = -1
                else:
                    if nxt not in index:
                        index[nxt] = len(states)
                        states.append(nxt)
                        queue.append(nxt)
                    j = index[nxt]
                row.append((p, j, cost, ABSORB_CODE[absorb]))
            rows.append(row)

        n = len(states)
        counts = np.array([len(r) for r in rows], dtype=np.int64)
        tr_ptr = np.zeros(n + 1, dtype=np.int64)
        tr_ptr[1:] = np.cumsum(counts)
        flat = [tr for r in rows for tr in r]
        tr_prob = np.array([f[0] for f in flat], dtype=float)
        tr_next = np.array([f[1] for f in flat], dtype=np.int64)
        tr_cost = np.array([f[2] for f in flat], dtype=np.int8)
        tr_absorb = np.array([f[3] for f in flat], dtype=np.int8)

        topo = _topological_order(n, tr_ptr, tr_next)
        max_cost_path = _max_cost_path(topo, tr_ptr, tr_next, tr_cost, root_idx)

        src = np.repeat(np.arange(n, dtype=np.int64), counts)
        to_state = tr_next >= 0
        m0 = (tr_cost == 0) & to_state
        m1 = (tr_cost == 1) & to_state
        Q0 = sparse.csr_matrix((tr_prob[m0], (src[m0], tr_next[m0])), shape=(n, n))
        Q1 = sparse.csr_matrix((tr_prob[m1], (src[m1], tr_next[m1])), shape=(n, n))

        def absorb_vector(cost: int, code: int) -> np.ndarray:
            mask = (tr_cost == cost) & (tr_absorb == code)
            v = np.zeros(n)
            np.add.at(v, src[mask], tr_prob[mask])
            return v

        paid_out = np.zeros(n)
        np.add.at(paid_out, src[tr_cost == 1], tr_prob[tr_cost == 1])

        return cls(
            states=states,
            index=index,
            roots=root_idx,
            tr_ptr=tr_ptr,
            tr_prob=tr_prob,
            tr_next=tr_next,
            tr_cost=tr_cost,
            tr_absorb=tr_absorb,
            Q0T=Q0.transpose().tocsr(),
            Q1T=Q1.transpose().tocsr(),
            a0_succ=absorb_vector(0, 1),
            a0_fail=absorb_vector(0, 2),
            a1_succ=absorb_vector(1, 1),
            a1_fail=absorb_vector(1, 2),
            paid_out=paid_out,
            topo=topo,
            max_cost_path=max_cost_path,
        )


def _topological_order(n: int, tr_ptr: np.ndarray, tr_next: np.ndarray) -> np.ndarray:
    ptr = tr_ptr.tolist()
    nxt = tr_next.tolist()
    indeg = [0] * n
    for j in nxt:
        if j >= 0:
            indeg[j] += 1
    ready = deque(i for i in range(n) if indeg[i] == 0)
    order: list[int] = []
    while ready:
        i = ready.popleft()
        order.append(i)
        for k in range(ptr[i], ptr[i + 1]):
            j = nxt[k]
            if j >= 0:
                indeg[j] -= 1
                if indeg[j] == 0:
                    ready.append(j)
    if len(order) != n:
        raise CycleError(
            "state graph has a cycle; every transition must strictly increase a counter"
        )
    return np.array(order, dtype=np.int64)


def _max_cost_path(
    topo: np.ndarray, tr_ptr: np.ndarray, tr_next: np.ndarray, tr_cost: np.ndarray, roots: list[int]
) -> int:
    ptr = tr_ptr.tolist()
    nxt = tr_next.tolist()
    cost = tr_cost.tolist()
    longest = [0] * len(topo)
    for i in reversed(topo.tolist()):
        best = 0
        for k in range(ptr[i], ptr[i + 1]):
            j = nxt[k]
            v = cost[k] + (longest[j] if j >= 0 else 0)
            if v > best:
                best = v
        longest[i] = best
    return max(longest[r] for r in roots)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_chain.py -v`
Expected: all PASS. (If `from tests.toy_models import ...` fails, add an empty `tests/__init__.py` and rerun.)

- [ ] **Step 6: Commit**

```bash
git add src/gacha/kernel tests/test_chain.py tests/toy_models.py tests/__init__.py
git commit -m "feat(kernel): transition types and DAG chain enumeration

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Forward algorithm (hitting-time distribution)

**Files:**
- Create: `src/gacha/kernel/forward.py`
- Test: `tests/test_forward.py`

**Interfaces:**
- Consumes: `EnumeratedChain` (Task 3).
- Produces: `hitting_time(chain, mu0=None, horizon=None) -> HittingTime` implementing spec §5.2.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forward.py
import numpy as np
import pytest

from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from tests.toy_models import ToyModel


def test_toy_hitting_time_exact():
    chain = EnumeratedChain.from_model(ToyModel())
    ht = hitting_time(chain)
    assert ht.f_succ.tolist() == pytest.approx([0.0, 0.6, 0.2])
    assert ht.f_fail.tolist() == pytest.approx([0.0, 0.2, 0.0])
    assert ht.residual == pytest.approx(0.0)
    assert ht.p_success == pytest.approx(0.8)


def test_mass_conservation():
    chain = EnumeratedChain.from_model(ToyModel())
    ht = hitting_time(chain)
    assert ht.f_succ.sum() + ht.f_fail.sum() + ht.residual == pytest.approx(1.0, abs=1e-12)


def test_truncated_horizon_reports_residual():
    chain = EnumeratedChain.from_model(ToyModel())
    ht = hitting_time(chain, horizon=1)
    assert ht.f_succ.tolist() == pytest.approx([0.0, 0.6])
    assert ht.f_fail.tolist() == pytest.approx([0.0, 0.2])
    assert ht.residual == pytest.approx(0.2)
    assert ht.residual_states == {"c": pytest.approx(0.2)}


def test_initial_distribution_over_states():
    chain = EnumeratedChain.from_model(ToyModel())
    mu0 = np.zeros(chain.n)
    mu0[chain.index["c"]] = 1.0
    ht = hitting_time(chain, mu0=mu0)
    assert ht.f_fail.tolist() == pytest.approx([0.5, 0.0, 0.0])
    assert ht.f_succ.tolist() == pytest.approx([0.0, 0.5, 0.0])


def test_rejects_bad_inputs():
    chain = EnumeratedChain.from_model(ToyModel())
    with pytest.raises(ValueError):
        hitting_time(chain, mu0=np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        hitting_time(chain, horizon=-1)
    multi = EnumeratedChain.from_model(ToyModel(), roots=["b", "c"])
    with pytest.raises(ValueError):
        hitting_time(multi)  # several roots: mu0 is required
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_forward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gacha.kernel.forward'`.

- [ ] **Step 3: Implement forward.py**

```python
# src/gacha/kernel/forward.py
"""Forward propagation of probability mass over paid-pull budget levels (spec §5.2)."""

from __future__ import annotations

import numpy as np

from .chain import CycleError, EnumeratedChain
from .types import HittingTime


def hitting_time(
    chain: EnumeratedChain, mu0: np.ndarray | None = None, horizon: int | None = None
) -> HittingTime:
    """Distribution of paid pulls until absorption.

    For each budget level j: take the closure of free (cost-0) moves, record absorptions that
    happen through free transitions at level j and through paid transitions at level j+1, then
    advance the remaining mass. The default horizon is the exact support bound
    ``chain.max_cost_path``; a smaller horizon leaves a positive residual.
    """
    if mu0 is None:
        if len(chain.roots) != 1:
            raise ValueError("pass mu0 explicitly when the chain has several roots")
        mu = chain.point_mass(chain.states[chain.roots[0]])
    else:
        mu = np.array(mu0, dtype=float)
        if mu.shape != (chain.n,) or (mu < 0).any() or abs(mu.sum() - 1.0) > 1e-9:
            raise ValueError("mu0 must be a probability vector over chain.states")
    horizon_j = chain.max_cost_path if horizon is None else int(horizon)
    if horizon_j < 0:
        raise ValueError("horizon must be >= 0")

    f_succ = np.zeros(horizon_j + 1)
    f_fail = np.zeros(horizon_j + 1)
    residual_measure = np.zeros(chain.n)
    for j in range(horizon_j + 1):
        w = _free_closure(chain, mu)
        f_succ[j] += w @ chain.a0_succ
        f_fail[j] += w @ chain.a0_fail
        if j < horizon_j:
            f_succ[j + 1] += w @ chain.a1_succ
            f_fail[j + 1] += w @ chain.a1_fail
            mu = chain.Q1T @ w
        else:
            residual_measure = w * chain.paid_out
    residual = float(residual_measure.sum())
    residual_states = {
        chain.states[i]: float(residual_measure[i]) for i in np.nonzero(residual_measure > 0)[0]
    }
    return HittingTime(f_succ, f_fail, residual, residual_states)


def _free_closure(chain: EnumeratedChain, mu: np.ndarray) -> np.ndarray:
    """W = sum_{i>=0} mu Q0^i; terminates because cost-0 transitions form a DAG."""
    w = mu.copy()
    v = mu
    for _ in range(chain.n + 1):
        v = chain.Q0T @ v
        if not v.any():
            return w
        w = w + v
    raise CycleError("free-transition closure did not terminate")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_forward.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gacha/kernel/forward.py tests/test_forward.py
git commit -m "feat(kernel): forward hitting-time distribution over paid pulls

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Backward algorithm (state values, success within budget)

**Files:**
- Create: `src/gacha/kernel/backward.py`
- Test: `tests/test_backward.py`

**Interfaces:**
- Consumes: `EnumeratedChain`, `hitting_time`.
- Produces: `StateValues(expectation, variance, p_success)` arrays indexed like `chain.states`; `state_values(chain) -> StateValues`; `success_within(chain, max_budget) -> np.ndarray` of shape `(max_budget+1, n)` with `S[b, i] = P(success with at most b paid pulls | start in state i)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_backward.py
import numpy as np
import pytest

from gacha.kernel.backward import state_values, success_within
from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from tests.toy_models import ToyModel


def test_toy_state_values():
    chain = EnumeratedChain.from_model(ToyModel())
    sv = state_values(chain)
    a, b, c = (chain.index[s] for s in "abc")
    assert sv.expectation[[a, b, c]].tolist() == pytest.approx([1.2, 1.2, 0.5])
    assert sv.variance[[a, b, c]].tolist() == pytest.approx([0.16, 0.16, 0.25])
    assert sv.p_success[[a, b, c]].tolist() == pytest.approx([0.8, 0.8, 0.5])


def test_backward_matches_forward_moments():
    chain = EnumeratedChain.from_model(ToyModel())
    sv = state_values(chain)
    ht = hitting_time(chain)
    j = np.arange(ht.horizon + 1)
    pmf = ht.pmf_stop
    mean = float((j * pmf).sum())
    var = float(((j - mean) ** 2 * pmf).sum())
    root = chain.roots[0]
    assert sv.expectation[root] == pytest.approx(mean, abs=1e-12)
    assert sv.variance[root] == pytest.approx(var, abs=1e-12)
    assert sv.p_success[root] == pytest.approx(ht.p_success, abs=1e-12)


def test_success_within_toy():
    chain = EnumeratedChain.from_model(ToyModel())
    s = success_within(chain, 2)
    a, b, c = (chain.index[x] for x in "abc")
    assert s[:, c].tolist() == pytest.approx([0.0, 0.5, 0.5])
    assert s[:, b].tolist() == pytest.approx([0.0, 0.6, 0.8])
    assert s[:, a].tolist() == pytest.approx([0.0, 0.6, 0.8])


def test_success_within_matches_forward_cdf():
    chain = EnumeratedChain.from_model(ToyModel())
    ht = hitting_time(chain)
    s = success_within(chain, ht.horizon)
    assert s[:, chain.roots[0]].tolist() == pytest.approx(np.cumsum(ht.f_succ).tolist())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gacha.kernel.backward'`.

- [ ] **Step 3: Implement backward.py**

```python
# src/gacha/kernel/backward.py
"""Backward (first-step) recurrences on the enumerated DAG (spec §5.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chain import EnumeratedChain


@dataclass
class StateValues:
    expectation: np.ndarray  # E[paid pulls until absorption | state]
    variance: np.ndarray
    p_success: np.ndarray


def state_values(chain: EnumeratedChain) -> StateValues:
    """E, Var and P(success) per state via reverse topological order.

    E[s]  = sum p (cost + E[next]);  M2[s] = sum p (cost^2 + 2 cost E[next] + M2[next]);
    Var = M2 - E^2;  P[s] = sum p (1{absorb=success} + P[next]).  Absorbed values are 0.
    """
    n = chain.n
    ptr = chain.tr_ptr.tolist()
    prob = chain.tr_prob.tolist()
    nxt = chain.tr_next.tolist()
    cost = chain.tr_cost.tolist()
    absorb = chain.tr_absorb.tolist()
    e = [0.0] * n
    m2 = [0.0] * n
    ps = [0.0] * n
    for i in reversed(chain.topo.tolist()):
        ei = m2i = pi = 0.0
        for k in range(ptr[i], ptr[i + 1]):
            p = prob[k]
            j = nxt[k]
            c = cost[k]
            if j >= 0:
                en, m2n, pn = e[j], m2[j], ps[j]
            else:
                en = m2n = 0.0
                pn = 1.0 if absorb[k] == 1 else 0.0
            ei += p * (c + en)
            m2i += p * (c * c + 2.0 * c * en + m2n)
            pi += p * pn
        e[i], m2[i], ps[i] = ei, m2i, pi
    expectation = np.array(e)
    variance = np.array(m2) - expectation**2
    return StateValues(expectation, variance, np.array(ps))


def success_within(chain: EnumeratedChain, max_budget: int) -> np.ndarray:
    """S[b, i] = P(success using at most b paid pulls | start in state i), b = 0..max_budget."""
    if max_budget < 0:
        raise ValueError("max_budget must be >= 0")
    n = chain.n
    ptr = chain.tr_ptr.tolist()
    prob = chain.tr_prob.tolist()
    nxt = chain.tr_next.tolist()
    cost = chain.tr_cost.tolist()
    absorb = chain.tr_absorb.tolist()
    s = np.zeros((max_budget + 1, n))
    for i in reversed(chain.topo.tolist()):
        acc = np.zeros(max_budget + 1)
        for k in range(ptr[i], ptr[i + 1]):
            p = prob[k]
            j = nxt[k]
            if j < 0:
                if absorb[k] == 1:
                    if cost[k] == 0:
                        acc += p
                    else:
                        acc[1:] += p
            elif cost[k] == 0:
                acc += p * s[:, j]
            else:
                acc[1:] += p * s[:-1, j]
        s[:, i] = acc
    return s
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backward.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gacha/kernel/backward.py tests/test_backward.py
git commit -m "feat(kernel): backward state values and success-within-budget DP

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Risk metrics and normal approximation

**Files:**
- Create: `src/gacha/risk/metrics.py`, `src/gacha/risk/normal.py`
- Test: `tests/test_risk.py`

**Interfaces:**
- Consumes: `HittingTime` (Task 3).
- Produces (all in `gacha.risk.metrics`, `which` is `"stop"` (default) or `"success"`, spec §5.8): `pmf_of(ht, which)`, `mean(ht, which)`, `var`, `sd`, `cv`, `cdf(ht, b, which)`, `survival(ht, b, which)` = P(T ≥ b), `completion(ht, b)` = P(success ∧ T ≤ b), `quantile(ht, alpha, which)`, `var_at` (alias), `cvar(ht, alpha, which)`, `expected_excess(ht, b, which)`, `skewness`, `entropy_nats`, `convolve(pmf, m) -> np.ndarray`, `from_pmf(pmf) -> HittingTime`, `summary(ht, alphas, budgets, which) -> pd.DataFrame` with columns `metric, value`. In `gacha.risk.normal`: `normal_approx_cdf(ht, b, which)`, `max_abs_cdf_error(ht, which)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_risk.py
import math

import numpy as np
import pytest

from gacha.kernel.types import HittingTime
from gacha.risk import metrics as rk
from gacha.risk.normal import max_abs_cdf_error, normal_approx_cdf


@pytest.fixture
def ht():
    # T in {1, 2, 3} with probabilities 0.5, 0.3, 0.2 (all success)
    return HittingTime(np.array([0.0, 0.5, 0.3, 0.2]), np.zeros(4), 0.0)


def test_moments(ht):
    assert rk.mean(ht) == pytest.approx(1.7)
    assert rk.var(ht) == pytest.approx(0.61)
    assert rk.sd(ht) == pytest.approx(math.sqrt(0.61))
    assert rk.cv(ht) == pytest.approx(math.sqrt(0.61) / 1.7)
    assert rk.skewness(ht) == pytest.approx(0.2760 / 0.61**1.5, abs=1e-3)
    assert rk.entropy_nats(ht) == pytest.approx(1.0297, abs=1e-4)


def test_cdf_survival_quantiles(ht):
    assert rk.cdf(ht, -1) == 0.0
    assert rk.cdf(ht, 1) == pytest.approx(0.5)
    assert rk.cdf(ht, 2) == pytest.approx(0.8)
    assert rk.cdf(ht, 99) == pytest.approx(1.0)
    assert rk.survival(ht, 2) == pytest.approx(0.5)
    assert rk.survival(ht, 0) == pytest.approx(1.0)
    assert rk.quantile(ht, 0.5) == 1
    assert rk.quantile(ht, 0.8) == 2
    assert rk.quantile(ht, 0.81) == 3
    assert rk.var_at(ht, 0.8) == 2
    with pytest.raises(ValueError):
        rk.quantile(ht, 0.0)
    with pytest.raises(ValueError):
        rk.quantile(ht, 1.5)


def test_cvar_and_expected_excess(ht):
    assert rk.cvar(ht, 0.8) == pytest.approx(2.4)
    assert rk.expected_excess(ht, 1) == pytest.approx(0.7)
    assert rk.expected_excess(ht, 3) == pytest.approx(0.0)


def test_defective_distribution_conventions():
    ht = HittingTime(np.array([0.0, 0.6, 0.2]), np.array([0.0, 0.2, 0.0]), 0.0)
    assert rk.mean(ht) == pytest.approx(1.2)  # stop distribution, mass 1
    assert rk.mean(ht, which="success") == pytest.approx(1.25)  # conditional on success
    assert rk.completion(ht, 1) == pytest.approx(0.6)
    assert rk.completion(ht, 2) == pytest.approx(0.8)
    assert rk.completion(ht, -3) == 0.0
    assert rk.cdf(ht, 1, which="success") == pytest.approx(0.75)


def test_truncated_stop_distribution_rejected():
    ht = HittingTime(np.array([0.0, 0.6]), np.array([0.0, 0.2]), 0.2)
    with pytest.raises(ValueError):
        rk.mean(ht)
    assert rk.mean(ht, which="success") == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rk.pmf_of(ht, "nonsense")


def test_convolve_and_from_pmf():
    assert rk.convolve(np.array([0.0, 0.5, 0.5]), 2).tolist() == pytest.approx([0, 0, 0.25, 0.5, 0.25])
    assert rk.convolve(np.array([0.0, 1.0]), 1).tolist() == pytest.approx([0.0, 1.0])
    h = rk.from_pmf(np.array([0.0, 0.25, 0.75]))
    assert h.p_success == pytest.approx(1.0)
    assert rk.mean(h) == pytest.approx(1.75)


def test_summary_dataframe(ht):
    df = rk.summary(ht, alphas=(0.5, 0.9), budgets=(1, 2))
    assert set(df.columns) == {"metric", "value"}
    rows = dict(zip(df["metric"], df["value"]))
    assert rows["mean"] == pytest.approx(1.7)
    assert rows["q50"] == 1
    assert rows["completion_at_2"] == pytest.approx(0.8)
    assert rows["expected_excess_at_1"] == pytest.approx(0.7)


def test_normal_approximation_on_binomial():
    pmf = rk.convolve(np.array([0.5, 0.5]), 40)  # Binomial(40, 0.5)
    h = rk.from_pmf(pmf)
    assert normal_approx_cdf(h, 20) == pytest.approx(rk.cdf(h, 20), abs=0.01)
    assert max_abs_cdf_error(h) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_risk.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gacha.risk.metrics'`.

- [ ] **Step 3: Implement metrics.py**

```python
# src/gacha/risk/metrics.py
"""Distributional risk metrics on HittingTime results (spec §5.8).

``which="stop"``: T_stop = paid pulls until absorption of either kind (proper distribution).
``which="success"``: success-only PMF, normalized, i.e. conditional on success.
``completion`` is the only metric that uses the unnormalized success mass.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from gacha.kernel.types import HittingTime

RESIDUAL_TOL = 1e-12


def pmf_of(ht: HittingTime, which: str = "stop") -> np.ndarray:
    if which == "stop":
        if ht.residual > RESIDUAL_TOL:
            raise ValueError(
                "stop distribution is truncated (residual > 0); recompute with a larger horizon"
            )
        return ht.pmf_stop
    if which == "success":
        return ht.f_succ
    raise ValueError(f"which must be 'stop' or 'success', got {which!r}")


def _normalized(ht: HittingTime, which: str) -> np.ndarray:
    pmf = pmf_of(ht, which)
    total = float(pmf.sum())
    if total <= 0.0:
        raise ValueError("distribution has zero mass")
    return pmf / total


def from_pmf(pmf: np.ndarray) -> HittingTime:
    """Wrap a plain PMF (e.g. a convolution) so every metric can be applied to it."""
    arr = np.asarray(pmf, dtype=float)
    return HittingTime(arr, np.zeros_like(arr), 0.0)


def convolve(pmf: np.ndarray, m: int) -> np.ndarray:
    """m-fold convolution of a PMF indexed by paid pulls (the paper's i.i.d. stages)."""
    if m < 1:
        raise ValueError("m must be >= 1")
    out = np.array([1.0])
    for _ in range(m):
        out = np.convolve(out, np.asarray(pmf, dtype=float))
    return out


def mean(ht: HittingTime, which: str = "stop") -> float:
    pmf = _normalized(ht, which)
    return float((np.arange(len(pmf)) * pmf).sum())


def var(ht: HittingTime, which: str = "stop") -> float:
    pmf = _normalized(ht, which)
    j = np.arange(len(pmf))
    m = float((j * pmf).sum())
    return float(((j - m) ** 2 * pmf).sum())


def sd(ht: HittingTime, which: str = "stop") -> float:
    return float(np.sqrt(var(ht, which)))


def cv(ht: HittingTime, which: str = "stop") -> float:
    return sd(ht, which) / mean(ht, which)


def cdf(ht: HittingTime, b: int, which: str = "stop") -> float:
    """P(T <= b)."""
    if b < 0:
        return 0.0
    pmf = _normalized(ht, which)
    return float(pmf[: min(b, len(pmf) - 1) + 1].sum())


def survival(ht: HittingTime, b: int, which: str = "stop") -> float:
    """P(T >= b), the paper's tail convention."""
    return 1.0 - cdf(ht, b - 1, which)


def completion(ht: HittingTime, b: int) -> float:
    """P(success and T <= b), unnormalized."""
    if b < 0:
        return 0.0
    return float(ht.f_succ[: min(b, ht.horizon) + 1].sum())


def quantile(ht: HittingTime, alpha: float, which: str = "stop") -> int:
    """Smallest j with P(T <= j) >= alpha."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    c = np.cumsum(_normalized(ht, which))
    return int(np.searchsorted(c, alpha - 1e-12, side="left"))


def var_at(ht: HittingTime, alpha: float, which: str = "stop") -> int:
    return quantile(ht, alpha, which)


def cvar(ht: HittingTime, alpha: float, which: str = "stop") -> float:
    """E[T | T >= VaR_alpha] (discrete upper-tail conditional mean, as in the paper)."""
    v = quantile(ht, alpha, which)
    pmf = _normalized(ht, which)
    tail = pmf[v:]
    j = np.arange(v, len(pmf))
    return float((j * tail).sum() / tail.sum())


def expected_excess(ht: HittingTime, b: int, which: str = "stop") -> float:
    """E[(T - b)^+]."""
    pmf = _normalized(ht, which)
    j = np.arange(len(pmf))
    return float(np.clip(j - b, 0, None) @ pmf)


def skewness(ht: HittingTime, which: str = "stop") -> float:
    pmf = _normalized(ht, which)
    j = np.arange(len(pmf))
    m = float((j * pmf).sum())
    s2 = float(((j - m) ** 2 * pmf).sum())
    m3 = float(((j - m) ** 3 * pmf).sum())
    return m3 / s2**1.5


def entropy_nats(ht: HittingTime, which: str = "stop") -> float:
    pmf = _normalized(ht, which)
    p = pmf[pmf > 0]
    return float(-(p * np.log(p)).sum())


def summary(
    ht: HittingTime,
    alphas: Sequence[float] = (0.5, 0.75, 0.9, 0.95, 0.99),
    budgets: Sequence[int] = (),
    which: str = "stop",
) -> pd.DataFrame:
    rows: list[tuple[str, float]] = [
        ("p_success", ht.p_success),
        ("mean", mean(ht, which)),
        ("sd", sd(ht, which)),
        ("skewness", skewness(ht, which)),
        ("entropy_nats", entropy_nats(ht, which)),
    ]
    for a in alphas:
        tag = f"{round(a * 100):d}"
        rows.append((f"q{tag}", quantile(ht, a, which)))
        rows.append((f"cvar{tag}", cvar(ht, a, which)))
    for b in budgets:
        rows.append((f"completion_at_{b}", completion(ht, b)))
        rows.append((f"expected_excess_at_{b}", expected_excess(ht, b, which)))
    return pd.DataFrame(rows, columns=["metric", "value"])
```

- [ ] **Step 4: Implement normal.py**

```python
# src/gacha/risk/normal.py
"""Continuity-corrected normal approximation diagnostics (paper §5.3)."""

from __future__ import annotations

from math import erf, sqrt

import numpy as np

from gacha.kernel.types import HittingTime

from . import metrics as rk


def _phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def normal_approx_cdf(ht: HittingTime, b: int, which: str = "stop") -> float:
    """Phi((b + 0.5 - mean) / sd)."""
    return _phi((b + 0.5 - rk.mean(ht, which)) / rk.sd(ht, which))


def max_abs_cdf_error(ht: HittingTime, which: str = "stop") -> float:
    pmf = rk.pmf_of(ht, which)
    c = np.cumsum(pmf / pmf.sum())
    return max(abs(float(c[j]) - normal_approx_cdf(ht, j, which)) for j in range(len(c)))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_risk.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gacha/risk tests/test_risk.py
git commit -m "feat(risk): distribution metrics, VaR/CVaR, convolution and normal diagnostics

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Paper models and reproduction tests

**Files:**
- Create: `src/gacha/models/paper.py`
- Test: `tests/test_paper_reproduction.py`

**Interfaces:**
- Consumes: `Transition`, `SUCCESS`, `FAIL` (Task 3); `PaperSchedule` (Task 2); kernel and risk.
- Produces: `SingleCounterModel(probs)` state `(t,)`; `Featured5050Model(probs, q=0.5, guarantee=False)` state `(t, g)`; `IIDStagesModel(model, m)` state `(stage, inner_state)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_paper_reproduction.py
"""Regression tests against Hou, Zhu & Zhang, Symmetry 2026, 18(6), 1051.

Tolerances (spec §6): 0.01 on moments, exact on integer quantiles, 5e-4 on probabilities.
A disagreement is investigated and documented in docs/research-notes.md, never forced.
"""

import pytest

from gacha.kernel.backward import state_values
from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.paper import Featured5050Model, IIDStagesModel, SingleCounterModel
from gacha.risk import metrics as rk
from gacha.risk.normal import max_abs_cdf_error
from gacha.rules.paper import PaperSchedule

MOM = 0.01
PROB = 5e-4


@pytest.fixture(scope="module")
def soft():
    chain = EnumeratedChain.from_model(SingleCounterModel(PaperSchedule().probs))
    return chain, hitting_time(chain)


@pytest.fixture(scope="module")
def t10(soft):
    return rk.from_pmf(rk.convolve(soft[1].pmf_stop, 10))


def test_single_stage_moments(soft):
    _, ht = soft
    assert ht.p_success == pytest.approx(1.0, abs=1e-12)
    assert rk.mean(ht) == pytest.approx(62.34, abs=MOM)
    assert rk.var(ht) == pytest.approx(592.43, abs=MOM)
    assert rk.sd(ht) == pytest.approx(24.34, abs=MOM)
    assert ht.horizon == 90


def test_expected_remaining_table(soft):
    chain, _ = soft
    sv = state_values(chain)
    expected = {0: 62.34, 70: 7.68, 72: 5.76, 73: 4.78, 80: 1.92, 89: 1.00}
    for t, value in expected.items():
        assert sv.expectation[chain.index[(t,)]] == pytest.approx(value, abs=MOM)


def test_variance_recurrence_matches_paper_form(soft):
    chain, _ = soft
    sv = state_values(chain)
    p = PaperSchedule().probs
    e = [sv.expectation[chain.index[(t,)]] for t in range(90)]
    v = [sv.variance[chain.index[(t,)]] for t in range(90)]
    for t in range(89):  # paper Eq. (10)
        assert v[t] == pytest.approx((1 - p[t]) * v[t + 1] + p[t] * (1 - p[t]) * e[t + 1] ** 2, abs=1e-9)
    assert v[89] == pytest.approx(0.0, abs=1e-12)


def test_hard_pity_only_baseline():
    ht = hitting_time(EnumeratedChain.from_model(SingleCounterModel(PaperSchedule().hard_only().probs)))
    assert rk.mean(ht) == pytest.approx(69.70, abs=MOM)
    assert rk.skewness(ht) == pytest.approx(-1.08, abs=MOM)
    assert rk.entropy_nats(ht) == pytest.approx(2.54, abs=MOM)


def test_single_stage_asymmetry(soft):
    _, ht = soft
    assert rk.skewness(ht) == pytest.approx(-1.24, abs=MOM)
    assert rk.entropy_nats(ht) == pytest.approx(3.61, abs=MOM)


def test_t10_moments_and_quantiles(t10):
    assert rk.mean(t10) == pytest.approx(623.38, abs=MOM)
    assert rk.var(t10) == pytest.approx(5924.34, abs=MOM)
    assert [rk.quantile(t10, a) for a in (0.5, 0.75, 0.9, 0.95, 0.99)] == [629, 679, 719, 741, 775]


def test_t10_tail_and_budget_metrics(t10):
    assert rk.survival(t10, 724) == pytest.approx(0.0876, abs=PROB)
    assert rk.survival(t10, 778) == pytest.approx(0.0073, abs=PROB)
    assert rk.cvar(t10, 0.9) == pytest.approx(744.66, abs=MOM)
    assert rk.expected_excess(t10, 700) == pytest.approx(5.11, abs=MOM)
    assert rk.completion(t10, 719) == pytest.approx(0.9014, abs=PROB)


def test_t10_via_iid_stages_model_equals_convolution(t10):
    ht = hitting_time(EnumeratedChain.from_model(IIDStagesModel(SingleCounterModel(PaperSchedule().probs), 10)))
    assert ht.pmf_stop.tolist() == pytest.approx(t10.pmf_stop.tolist(), abs=1e-12)


def test_normal_approximation_error(t10):
    err = max_abs_cdf_error(t10)
    assert 0.02 < err < 0.04  # paper: "approximately 0.03"


def test_featured_target_no_guarantee():
    ht = hitting_time(EnumeratedChain.from_model(Featured5050Model(PaperSchedule().probs)))
    assert ht.horizon == 180
    assert rk.mean(ht) == pytest.approx(93.51, abs=MOM)
    assert rk.sd(ht) == pytest.approx(43.13, abs=MOM)
    assert [rk.quantile(ht, a) for a in (0.9, 0.95, 0.99)] == [156, 158, 161]
    assert rk.skewness(ht) == pytest.approx(0.01, abs=MOM)
    assert rk.entropy_nats(ht) == pytest.approx(4.54, abs=MOM)


def test_featured_target_with_guarantee_equals_single_stage(soft):
    ht = hitting_time(EnumeratedChain.from_model(Featured5050Model(PaperSchedule().probs, guarantee=True)))
    assert ht.pmf_stop.tolist() == pytest.approx(soft[1].pmf_stop.tolist(), abs=1e-12)
    assert [rk.quantile(ht, a) for a in (0.9, 0.95, 0.99)] == [80, 81, 83]


@pytest.mark.parametrize(
    "m, mean, sd, q90, q95",
    [(2, 187.01, 60.99, 265, 300), (5, 467.53, 96.44, 593, 626), (10, 935.06, 136.39, 1111, 1160)],
)
def test_repeated_featured_targets(m, mean, sd, q90, q95):
    ht = hitting_time(EnumeratedChain.from_model(IIDStagesModel(Featured5050Model(PaperSchedule().probs), m)))
    assert rk.mean(ht) == pytest.approx(mean, abs=MOM)
    assert rk.sd(ht) == pytest.approx(sd, abs=MOM)
    assert rk.quantile(ht, 0.9) == q90
    assert rk.quantile(ht, 0.95) == q95
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_paper_reproduction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gacha.models.paper'`.

- [ ] **Step 3: Implement paper.py**

```python
# src/gacha/models/paper.py
"""Models from Hou, Zhu & Zhang (2026): single pity counter, featured 50/50, i.i.d. stages."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

from gacha.kernel.types import FAIL, SUCCESS, Model, Transition


class SingleCounterModel:
    """State (t,), t = consecutive failures. Success with p_t, else t+1. Paper §2-3."""

    def __init__(self, probs: Sequence[float]):
        self.p = tuple(float(x) for x in probs)
        if not self.p or self.p[-1] != 1.0:
            raise ValueError("the last probability must be 1 (hard pity)")

    def initial(self) -> Hashable:
        return (0,)

    def step(self, s: Hashable) -> list[Transition]:
        (t,) = s
        p = self.p[t]
        out = [Transition(p, None, cost=1, absorb=SUCCESS)]
        if p < 1.0:
            out.append(Transition(1.0 - p, (t + 1,), cost=1))
        return out


class Featured5050Model:
    """State (t, g): g=1 means the next rare item is guaranteed featured. Paper Eq. (21)-(22)."""

    def __init__(self, probs: Sequence[float], q: float = 0.5, guarantee: bool = False):
        self.p = tuple(float(x) for x in probs)
        if not self.p or self.p[-1] != 1.0:
            raise ValueError("the last probability must be 1 (hard pity)")
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be in [0, 1]")
        self.q = q
        self.guarantee = guarantee

    def initial(self) -> Hashable:
        return (0, 1 if self.guarantee else 0)

    def step(self, s: Hashable) -> list[Transition]:
        t, g = s
        p = self.p[t]
        out: list[Transition] = []
        if g == 1:
            out.append(Transition(p, None, cost=1, absorb=SUCCESS))
        else:
            out.append(Transition(p * self.q, None, cost=1, absorb=SUCCESS))
            out.append(Transition(p * (1.0 - self.q), (0, 1), cost=1))
        if p < 1.0:
            out.append(Transition(1.0 - p, (t + 1, g), cost=1))
        return out


class IIDStagesModel:
    """m independent repetitions of ``model``; each success restarts from model.initial()."""

    def __init__(self, model: Model, m: int):
        if m < 1:
            raise ValueError("m must be >= 1")
        self.model = model
        self.m = m

    def initial(self) -> Hashable:
        return (0, self.model.initial())

    def step(self, s: Hashable) -> list[Transition]:
        stage, inner = s
        out: list[Transition] = []
        for tr in self.model.step(inner):
            if tr.absorb == SUCCESS:
                if stage + 1 == self.m:
                    out.append(Transition(tr.prob, None, cost=tr.cost, absorb=SUCCESS))
                else:
                    out.append(Transition(tr.prob, (stage + 1, self.model.initial()), cost=tr.cost))
            elif tr.absorb == FAIL:
                out.append(tr)
            else:
                out.append(Transition(tr.prob, (stage, tr.next), cost=tr.cost))
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_paper_reproduction.py -v`
Expected: all PASS. If a single value is off (for example a quantile by one), do **not** edit the expected number: check the quantile convention (`P(T <= j) >= alpha`), re-read the paper table, and if the paper is inconsistent record the finding in `docs/research-notes.md` (Task 15 creates the file; create it early with a "Reproduction notes" section if needed) and mark that assertion with a comment citing the computed value.

- [ ] **Step 5: Commit**

```bash
git add src/gacha/models/paper.py tests/test_paper_reproduction.py
git commit -m "feat(models): paper single-counter, featured 50/50 and i.i.d. stage models with reproduction tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Endfield single-banner model

**Files:**
- Create: `src/gacha/models/endfield.py`
- Test: `tests/test_endfield_model.py`

**Interfaces:**
- Consumes: `EndfieldCharacterRules`, `BannerSpec` (Task 2); kernel types; `state_values`, `hitting_time`.
- Produces: `BannerState(t, n, c, u)` NamedTuple; `pull(rules, state, f, target_copies) -> list[tuple[float, BannerState, int]]` (prob, next, cost); `status(rules, state, spec, f) -> "active" | "success" | "exhausted"`; `validate_start(spec, start, f)`; `SingleBannerModel(spec, start=BannerState(0,0,0,0), dossier=False)` with attributes `.spec`, `.rules`, `.f`, `.start`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_endfield_model.py
from dataclasses import replace

import numpy as np
import pytest

from gacha.kernel.backward import state_values
from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.endfield import BannerState, SingleBannerModel, pull, status, validate_start
from gacha.risk import metrics as rk
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules

FULL = EndfieldCharacterRules()
# Only the 80-pull pity and the 50/50 (R2-R4); everything banner-local switched off.
BARE = EndfieldCharacterRules(
    guarantee_pull=None, vacuum_at=None, dossier_at=None, potential_every=None, free_start_pulls=0
)


def ht_of(spec, **kw):
    return hitting_time(EnumeratedChain.from_model(SingleBannerModel(spec, **kw)))


def test_pull_branches_and_costs():
    f = FULL.free_pulls(False)  # 5
    trs = pull(FULL, BannerState(0, 0, 0, 0), f, 1)
    assert all(cost == 0 for _, _, cost in trs)  # first pull is free
    probs = {nxt: p for p, nxt, _ in trs}
    assert probs[BannerState(0, 1, 1, 1)] == pytest.approx(0.004)  # UP
    assert probs[BannerState(0, 1, 0, 0)] == pytest.approx(0.004)  # off-rate 6*
    assert probs[BannerState(1, 1, 0, 0)] == pytest.approx(0.992)
    trs = pull(FULL, BannerState(10, 5, 0, 0), f, 1)
    assert all(cost == 1 for _, _, cost in trs)  # sixth counted pull is paid


def test_pull_guarantee_forces_up_once():
    trs = pull(BARE, BannerState(3, 119, 0, 0), 0, 1)
    assert len(trs) == 3  # guarantee disabled in BARE
    rules = replace(BARE, guarantee_pull=120)
    trs = pull(rules, BannerState(3, 119, 0, 0), 0, 1)
    assert trs == [(1.0, BannerState(0, 120, 1, 1), 1)]
    trs = pull(rules, BannerState(3, 119, 0, 1), 0, 1)  # already obtained: no force
    assert sum(p for p, nxt, _ in trs if nxt.c == 1) == pytest.approx(0.004)


def test_pull_vacuum_and_potential_bonus():
    rules = replace(BARE, vacuum_at=30)
    trs = pull(rules, BannerState(2, 29, 0, 0), 0, 1)
    p_vac = 1 - (1 - 0.004) ** 10  # P(at least one UP among the 10 vacuum pulls)
    reached = sum(p for p, nxt, _ in trs if nxt.c == 1)
    assert reached == pytest.approx(0.004 + 0.996 * p_vac)
    rules = replace(BARE, potential_every=240)
    trs = pull(rules, BannerState(0, 239, 1, 1), 0, 2)
    assert all(nxt.c == 2 for _, nxt, _ in trs)


def test_status_rules():
    spec = BannerSpec(FULL, target_copies=1, cap=10)
    assert status(FULL, BannerState(0, 0, 0, 0), spec, 5) == "active"
    assert status(FULL, BannerState(0, 3, 1, 1), spec, 5) == "active"  # free pulls remain (P3)
    assert status(FULL, BannerState(0, 5, 1, 1), spec, 5) == "success"
    assert status(FULL, BannerState(0, 15, 0, 0), spec, 5) == "exhausted"  # paid 10 == cap
    skip = BannerSpec(FULL, target_copies=0, cap=0)
    assert status(FULL, BannerState(0, 4, 0, 0), skip, 5) == "active"
    assert status(FULL, BannerState(0, 5, 0, 0), skip, 5) == "exhausted"


def test_validate_start_rejects_bad_states():
    spec = BannerSpec(FULL, 1, 120)
    validate_start(spec, BannerState(79, 0, 0, 0), 5)
    with pytest.raises(ValueError):
        validate_start(spec, BannerState(80, 0, 0, 0), 5)
    with pytest.raises(ValueError):
        validate_start(spec, BannerState(0, 0, 2, 1), 5)  # more copies than target
    with pytest.raises(ValueError):
        validate_start(spec, BannerState(0, 200, 0, 0), 5)  # beyond paid cap
    with pytest.raises(ValueError):
        validate_start(spec, BannerState(0, 0, 0, 2), 5)
    with pytest.raises(ValueError):
        SingleBannerModel(spec, start=BannerState(-1, 0, 0, 0))


def test_first_six_star_closed_form():
    # up_share=1: first UP == first 6*, bounded by hard pity, E = sum_{j<80} (1-p)^j
    rules = replace(BARE, up_share=1.0)
    ht = ht_of(BannerSpec(rules, 1, 80))
    assert ht.p_success == pytest.approx(1.0, abs=1e-12)
    assert rk.mean(ht) == pytest.approx(sum(0.992**j for j in range(80)), abs=1e-9)
    assert ht.f_succ[80] == pytest.approx(0.992**79, abs=1e-12)


def test_fifty_fifty_closed_form():
    # hard_pity=1: every pull is a 6*, so T ~ Geometric(0.5) truncated at the cap
    rules = replace(BARE, hard_pity=1, soft_pity_start=0, soft_pity_step=0.0)
    cap = 6
    ht = ht_of(BannerSpec(rules, 1, cap))
    expected = [0.0] + [0.5**j for j in range(1, cap + 1)]
    assert ht.f_succ.tolist() == pytest.approx(expected, abs=1e-12)
    assert ht.p_fail == pytest.approx(0.5**cap)
    assert rk.mean(ht) == pytest.approx(sum(j * 0.5**j for j in range(1, cap + 1)) + cap * 0.5**cap)


def test_support_bound_from_guarantee():
    ht = ht_of(BannerSpec(replace(FULL, free_start_pulls=0), 1, 120))
    assert ht.horizon == 120 and ht.f_succ[120] > 0 and ht.p_success == pytest.approx(1.0)
    ht = ht_of(BannerSpec(FULL, 1, 120), dossier=True)  # f = 15
    assert ht.horizon == 105 and ht.f_succ[105] > 0 and ht.p_success == pytest.approx(1.0)


def test_entering_pity_79_gives_immediate_coin_flip():
    ht = ht_of(BannerSpec(BARE, 1, 120), start=BannerState(79, 0, 0, 0))
    assert ht.f_succ[1] == pytest.approx(0.5)


def test_guarantee_void_after_up():
    rules = replace(BARE, guarantee_pull=120)
    voided = ht_of(BannerSpec(rules, 2, 200), start=BannerState(0, 119, 1, 1))
    assert voided.f_succ[1] == pytest.approx(0.004)
    live = ht_of(BannerSpec(rules, 1, 200), start=BannerState(0, 119, 0, 0))
    assert live.f_succ[1] == pytest.approx(1.0)


def test_vacuum_adds_exact_success_mass_at_pull_30():
    without = ht_of(BannerSpec(BARE, 1, 40))
    with_v = ht_of(BannerSpec(replace(BARE, vacuum_at=30), 1, 40))
    survive = 1.0 - without.f_succ[:30].sum()
    p_vac = 1 - (1 - 0.004) ** 10
    assert with_v.f_succ[:30].tolist() == pytest.approx(without.f_succ[:30].tolist(), abs=1e-12)
    assert with_v.f_succ[30] == pytest.approx(without.f_succ[30] + (survive - without.f_succ[30]) * p_vac, abs=1e-12)


def test_potential_at_240_completes_second_copy():
    rules = replace(BARE, potential_every=240)
    ht = ht_of(BannerSpec(rules, 2, 300), start=BannerState(0, 239, 1, 1))
    assert ht.f_succ[1] == pytest.approx(1.0)


def test_expected_pulls_monotone_in_entering_pity():
    """Spec §6: expected to hold; a failure is a research finding, not a bug to silence."""
    spec = BannerSpec(FULL, 1, 120)
    model = SingleBannerModel(spec)
    roots = [BannerState(t, 0, 0, 0) for t in range(FULL.hard_pity)]
    chain = EnumeratedChain.from_model(model, roots=roots)
    sv = state_values(chain)
    e = [sv.expectation[chain.index[r]] for r in roots]
    assert all(e[t + 1] <= e[t] + 1e-9 for t in range(len(e) - 1))


def test_golden_default_rules():
    """Frozen after first computation (Step 5 below). Update only with a justification."""
    ht = ht_of(BannerSpec(FULL, 1, 120))
    assert rk.mean(ht) == pytest.approx(GOLDEN["mean"], rel=1e-9)
    assert rk.sd(ht) == pytest.approx(GOLDEN["sd"], rel=1e-9)
    assert rk.quantile(ht, 0.5) == GOLDEN["q50"]
    assert rk.quantile(ht, 0.9) == GOLDEN["q90"]
    assert rk.quantile(ht, 0.95) == GOLDEN["q95"]
    assert ht.f_succ[115] == pytest.approx(GOLDEN["p_115"], rel=1e-9)


GOLDEN = {"mean": None, "sd": None, "q50": None, "q90": None, "q95": None, "p_115": None}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_endfield_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gacha.models.endfield'`.

- [ ] **Step 3: Implement endfield.py**

```python
# src/gacha/models/endfield.py
"""Endfield limited character banner as a finite DAG model (spec §5.5)."""

from __future__ import annotations

from collections.abc import Hashable
from typing import NamedTuple

from gacha.kernel.types import FAIL, SUCCESS, Transition
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules

ACTIVE = "active"
DONE = "success"
EXHAUSTED = "exhausted"


class BannerState(NamedTuple):
    t: int  # pity counter (carried across banners)
    n: int  # counted pulls in this banner, free and paid
    c: int  # UP copies obtained in this banner, capped at target_copies
    u: int  # 1 once a UP came from a counted pull (voids the 120 guarantee)


def pull(
    rules: EndfieldCharacterRules, state: BannerState, f: int, target_copies: int
) -> list[tuple[float, BannerState, int]]:
    """Distribution of the next state after one counted pull: (prob, next, cost)."""
    t, n, c, u = state
    cost = 0 if n < f else 1
    n1 = n + 1
    branches: list[tuple[float, int, int, int]] = []  # (prob, t', c', u')
    if rules.guarantee_pull is not None and u == 0 and n1 == rules.guarantee_pull:
        branches.append((1.0, 0, c + 1, 1))  # R5: forced UP, resets pity
    else:
        p = rules.p(t)
        q = rules.up_share
        if p * q > 0.0:
            branches.append((p * q, 0, c + 1, 1))  # R4: UP
        if p * (1.0 - q) > 0.0:
            branches.append((p * (1.0 - q), 0, c, u))  # off-rate 6*, R3 reset
        if p < 1.0:
            branches.append((1.0 - p, t + 1, c, u))
    bonus = 1 if (rules.potential_every is not None and n1 % rules.potential_every == 0) else 0  # R8
    vacuum = rules.vacuum_copies_pmf() if (rules.vacuum_at is not None and n1 == rules.vacuum_at) else (1.0,)
    merged: dict[BannerState, float] = {}
    for pr, t1, c1, u1 in branches:
        for k, pk in enumerate(vacuum):
            if pk == 0.0:
                continue
            c2 = min(c1 + bonus + k, target_copies)
            key = BannerState(t1, n1, c2, u1)
            merged[key] = merged.get(key, 0.0) + pr * pk
    return [(pr, st, cost) for st, pr in merged.items()]


def status(rules: EndfieldCharacterRules, state: BannerState, spec: BannerSpec, f: int) -> str:
    """P3: success once the target is reached and free pulls are used up; exhausted at the cap."""
    _, n, c, _ = state
    reached = spec.target_copies >= 1 and c >= spec.target_copies
    if reached and n >= f:
        return DONE
    if not reached and n >= f and max(0, n - f) >= spec.cap:
        return EXHAUSTED  # free pulls are always used first (P1, P2), then the cap applies
    return ACTIVE


def validate_start(spec: BannerSpec, start: BannerState, f: int) -> None:
    t, n, c, u = start
    if not 0 <= t < spec.rules.hard_pity:
        raise ValueError(f"pity t must be in [0, {spec.rules.hard_pity}), got {t}")
    if n < 0:
        raise ValueError("banner pulls n must be >= 0")
    if not 0 <= c <= spec.target_copies:
        raise ValueError(f"copies c must be in [0, {spec.target_copies}], got {c}")
    if u not in (0, 1):
        raise ValueError("u must be 0 or 1")
    if max(0, n - f) > spec.cap:
        raise ValueError(f"start state has {max(0, n - f)} paid pulls, beyond the cap {spec.cap}")


class SingleBannerModel:
    """One banner; absorbs success when the target is reached, fail when the cap is exhausted."""

    def __init__(
        self, spec: BannerSpec, start: BannerState = BannerState(0, 0, 0, 0), dossier: bool = False
    ):
        self.spec = spec
        self.rules = spec.rules
        self.f = spec.rules.free_pulls(dossier)
        self.start = BannerState(*start)
        validate_start(spec, self.start, self.f)

    def initial(self) -> Hashable:
        return self.start

    def step(self, s: Hashable) -> list[Transition]:
        st = status(self.rules, s, self.spec, self.f)
        if st == DONE:
            return [Transition(1.0, None, cost=0, absorb=SUCCESS)]
        if st == EXHAUSTED:
            return [Transition(1.0, None, cost=0, absorb=FAIL)]
        return [
            Transition(pr, nxt, cost=cost)
            for pr, nxt, cost in pull(self.rules, s, self.f, self.spec.target_copies)
        ]
```

- [ ] **Step 4: Run the tests except the golden one**

Run: `uv run pytest tests/test_endfield_model.py -v -k "not golden"`
Expected: all PASS. If `test_expected_pulls_monotone_in_entering_pity` fails, keep the test, print the offending `t` values, and write the finding into `docs/research-notes.md` under "Findings" (create the file with that heading if Task 15 has not run yet); then mark the test `xfail(strict=True, reason="documented finding")`.

- [ ] **Step 5: Freeze the golden values**

Run:
```bash
uv run python -c "
from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.endfield import SingleBannerModel
from gacha.risk import metrics as rk
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules
ht = hitting_time(EnumeratedChain.from_model(SingleBannerModel(BannerSpec(EndfieldCharacterRules(), 1, 120))))
print({'mean': rk.mean(ht), 'sd': rk.sd(ht), 'q50': rk.quantile(ht, .5), 'q90': rk.quantile(ht, .9), 'q95': rk.quantile(ht, .95), 'p_115': float(ht.f_succ[115])})
"
```
Paste the printed dict into `GOLDEN` in `tests/test_endfield_model.py` with a comment `# frozen 2026-09-23, first UP, default rules, f=5, cap 120`. Sanity: mean must lie between 40 and 90, `q95` must equal 115 or less, `p_115` must be between 0.01 and 0.3.

Run: `uv run pytest tests/test_endfield_model.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gacha/models/endfield.py tests/test_endfield_model.py
git commit -m "feat(models): Endfield single-banner model with guarantee, vacuum, dossier and potential rules

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Multi-banner plan model

**Files:**
- Create: `src/gacha/models/plan.py`
- Test: `tests/test_plan.py`

**Interfaces:**
- Consumes: `BannerState`, `pull`, `status`, `validate_start`, `ACTIVE/DONE/EXHAUSTED` (Task 8); `BannerSpec`.
- Produces: `PlanState(k, t, n, c, u, d)` NamedTuple; `Plan(banners, start=BannerState(0,0,0,0), dossier0=False)` model with `.banners`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_plan.py
import numpy as np
import pytest

from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.kernel.types import FAIL, SUCCESS
from gacha.models.endfield import BannerState, SingleBannerModel
from gacha.models.plan import Plan, PlanState
from gacha.risk import metrics as rk
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules

FULL = EndfieldCharacterRules()
WANTED = BannerSpec(FULL, 1, 120)
SKIP = BannerSpec(FULL, 0, 0)


def ht_of(model):
    return hitting_time(EnumeratedChain.from_model(model))


def test_plan_rejects_empty():
    with pytest.raises(ValueError):
        Plan([])


def test_one_banner_plan_equals_single_banner():
    single = ht_of(SingleBannerModel(WANTED))
    plan = ht_of(Plan([WANTED]))
    assert plan.f_succ.tolist() == pytest.approx(single.f_succ.tolist(), abs=1e-12)
    assert plan.f_fail.tolist() == pytest.approx(single.f_fail.tolist(), abs=1e-12)


def test_boundary_transitions():
    plan = Plan([WANTED, WANTED])
    # banner 0 finished with n >= 60: dossier granted to banner 1
    [tr] = plan.step(PlanState(0, 3, 61, 1, 1, 0))
    assert tr.next == PlanState(1, 3, 0, 0, 0, 1) and tr.cost == 0 and tr.prob == 1.0
    [tr] = plan.step(PlanState(0, 3, 59, 1, 1, 0))
    assert tr.next == PlanState(1, 3, 0, 0, 0, 0)
    # wanted banner exhausted -> fail; last banner success -> success
    [tr] = plan.step(PlanState(0, 3, 125, 0, 0, 0))
    assert tr.absorb == FAIL
    [tr] = plan.step(PlanState(1, 3, 20, 1, 1, 1))
    assert tr.absorb == SUCCESS


def test_skipped_banner_only_uses_free_pulls():
    ht = ht_of(Plan([SKIP, WANTED]))
    # success with 0 paid pulls needs a UP among banner 2's 5 free pulls; banner 1's free
    # pulls only move the pity counter inside the flat region, so the probability is flat.
    assert ht.f_succ[0] == pytest.approx(1 - (1 - 0.004) ** 5, abs=1e-12)
    assert ht.horizon == 115


def test_plan_last_skipped_banner_succeeds():
    ht = ht_of(Plan([WANTED, SKIP]))
    assert ht.p_success == pytest.approx(1.0, abs=1e-12)
    assert ht.horizon == 115


def test_dossier_gives_next_banner_fifteen_free_pulls():
    # start banner 0 already at n=60 (dossier earned) with the UP in hand: banner 1 gets f=15
    plan = Plan([WANTED, WANTED], start=BannerState(0, 60, 1, 1))
    ht = ht_of(plan)
    fresh15 = ht_of(SingleBannerModel(WANTED, dossier=True))
    assert ht.horizon == fresh15.horizon == 105
    assert ht.f_succ.tolist() == pytest.approx(fresh15.f_succ.tolist(), abs=1e-12)


def test_cross_banner_coupling_differs_from_iid_convolution():
    single = ht_of(SingleBannerModel(WANTED))
    exact = ht_of(Plan([WANTED, WANTED]))
    conv = rk.from_pmf(rk.convolve(single.pmf_stop, 2))
    n = max(exact.horizon, conv.horizon) + 1
    ce = np.cumsum(np.pad(exact.pmf_stop, (0, n - exact.horizon - 1)))
    cc = np.cumsum(np.pad(conv.pmf_stop, (0, n - conv.horizon - 1)))
    assert np.abs(ce - cc).max() > 1e-3
    assert exact.p_success == pytest.approx(1.0, abs=1e-12)


def test_validates_start_against_first_banner():
    with pytest.raises(ValueError):
        Plan([WANTED], start=BannerState(0, 0, 3, 1))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gacha.models.plan'`.

- [ ] **Step 3: Implement plan.py**

```python
# src/gacha/models/plan.py
"""A chronological sequence of banners with carried pity and dossier (spec §5.6)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import NamedTuple

from gacha.kernel.types import FAIL, SUCCESS, Transition
from gacha.rules.endfield import BannerSpec

from .endfield import ACTIVE, DONE, BannerState, pull, status, validate_start


class PlanState(NamedTuple):
    k: int  # banner index
    t: int
    n: int
    c: int
    u: int
    d: int  # 1 if a dossier (10 free pulls) entered this banner


class Plan:
    """Player policy: pull each wanted banner until its target or its cap; skipped banners
    only use free pulls (P2). A wanted banner that exhausts its cap fails the whole plan."""

    def __init__(
        self,
        banners: Sequence[BannerSpec],
        start: BannerState = BannerState(0, 0, 0, 0),
        dossier0: bool = False,
    ):
        self.banners = list(banners)
        if not self.banners:
            raise ValueError("a plan needs at least one banner")
        self.start = BannerState(*start)
        self.dossier0 = 1 if dossier0 else 0
        first = self.banners[0]
        validate_start(first, self.start, first.rules.free_pulls(bool(self.dossier0)))

    def initial(self) -> Hashable:
        return PlanState(0, *self.start, self.dossier0)

    def step(self, s: Hashable) -> list[Transition]:
        k, t, n, c, u, d = s
        spec = self.banners[k]
        rules = spec.rules
        f = rules.free_pulls(bool(d))
        bs = BannerState(t, n, c, u)
        st = status(rules, bs, spec, f)
        if st == ACTIVE:
            return [
                Transition(pr, PlanState(k, *nxt, d), cost=cost)
                for pr, nxt, cost in pull(rules, bs, f, spec.target_copies)
            ]
        if st != DONE and spec.target_copies >= 1:
            return [Transition(1.0, None, cost=0, absorb=FAIL)]
        if k == len(self.banners) - 1:
            return [Transition(1.0, None, cost=0, absorb=SUCCESS)]
        d1 = 1 if (rules.dossier_at is not None and n >= rules.dossier_at) else 0  # R7
        return [Transition(1.0, PlanState(k + 1, t, 0, 0, 0, d1), cost=0)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plan.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gacha/models/plan.py tests/test_plan.py
git commit -m "feat(models): multi-banner plan with pity carry-over and dossier propagation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Monte Carlo simulator and parity checks

**Files:**
- Create: `src/gacha/mc/simulate.py`, `src/gacha/mc/compare.py`
- Test: `tests/test_mc.py`

**Interfaces:**
- Consumes: `EnumeratedChain` arrays (Task 3), risk metrics (Task 6).
- Produces: `Paths(absorb: np.ndarray[int8], paid: np.ndarray[int64])`; `simulate(chain, n_paths, seed, root=None) -> Paths`; `compare(ht, paths, thresholds=(), tol_se=3.0) -> pd.DataFrame` with columns `metric, exact, mc, se, z, ok`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mc.py
import numpy as np
import pytest

from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.mc.compare import compare
from gacha.mc.simulate import simulate
from gacha.models.endfield import SingleBannerModel
from gacha.models.paper import SingleCounterModel
from gacha.models.plan import Plan
from gacha.risk import metrics as rk
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules
from gacha.rules.paper import PaperSchedule
from tests.toy_models import ToyModel

N = 200_000
SEED = 20260923


def test_toy_paths_have_expected_shape_and_frequencies():
    chain = EnumeratedChain.from_model(ToyModel())
    paths = simulate(chain, 50_000, seed=1)
    assert paths.absorb.shape == (50_000,) and paths.paid.shape == (50_000,)
    assert set(np.unique(paths.absorb).tolist()) <= {1, 2}
    assert (paths.absorb == 1).mean() == pytest.approx(0.8, abs=0.01)
    assert paths.paid.mean() == pytest.approx(1.2, abs=0.01)


def test_simulation_is_reproducible():
    chain = EnumeratedChain.from_model(ToyModel())
    a = simulate(chain, 1000, seed=7)
    b = simulate(chain, 1000, seed=7)
    assert np.array_equal(a.paid, b.paid) and np.array_equal(a.absorb, b.absorb)


def test_compare_flags_agreement():
    chain = EnumeratedChain.from_model(ToyModel())
    df = compare(hitting_time(chain), simulate(chain, N, seed=SEED), thresholds=(1, 2))
    assert set(df["metric"]) == {"p_success", "mean", "survival_at_1", "survival_at_2"}
    assert df["ok"].all()


def _parity(chain, thresholds):
    df = compare(hitting_time(chain), simulate(chain, N, seed=SEED), thresholds=thresholds)
    assert df["ok"].all(), df.to_string()


def test_parity_paper_single_counter():
    _parity(EnumeratedChain.from_model(SingleCounterModel(PaperSchedule().probs)), (80,))


def test_parity_endfield_first_up():
    spec = BannerSpec(EndfieldCharacterRules(), 1, 120)
    _parity(EnumeratedChain.from_model(SingleBannerModel(spec)), (60, 100))


def test_parity_two_banner_plan():
    spec = BannerSpec(EndfieldCharacterRules(), 1, 120)
    _parity(EnumeratedChain.from_model(Plan([spec, spec])), (120, 180))


def test_simulate_requires_root_for_multi_root_chain():
    chain = EnumeratedChain.from_model(ToyModel(), roots=["b", "c"])
    with pytest.raises(ValueError):
        simulate(chain, 10, seed=0)
    paths = simulate(chain, 10, seed=0, root="c")
    assert paths.paid.max() <= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gacha.mc.simulate'`.

- [ ] **Step 3: Implement simulate.py**

```python
# src/gacha/mc/simulate.py
"""Vectorized Monte Carlo on the enumerated chain; samples the same transitions the exact
engine uses, so rules are never re-implemented here (spec §5.7)."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

from gacha.kernel.chain import EnumeratedChain


@dataclass
class Paths:
    absorb: np.ndarray  # int8: 1 success, 2 fail
    paid: np.ndarray  # int64 paid pulls per path


def _tables(chain: EnumeratedChain) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dense per-state tables padded to the maximum out-degree.

    Padded cumulative probabilities are 1.0, so a uniform draw u < 1 never selects a pad."""
    n = chain.n
    deg = np.diff(chain.tr_ptr)
    width = int(deg.max())
    cum = np.ones((n, width))
    nxt = np.full((n, width), -1, dtype=np.int64)
    cost = np.zeros((n, width), dtype=np.int8)
    absorb = np.zeros((n, width), dtype=np.int8)
    ptr = chain.tr_ptr
    for i in range(n):
        a, b = int(ptr[i]), int(ptr[i + 1])
        k = b - a
        c = np.cumsum(chain.tr_prob[a:b])
        c[-1] = 1.0
        cum[i, :k] = c
        nxt[i, :k] = chain.tr_next[a:b]
        cost[i, :k] = chain.tr_cost[a:b]
        absorb[i, :k] = chain.tr_absorb[a:b]
    return cum, nxt, cost, absorb


def simulate(
    chain: EnumeratedChain, n_paths: int, seed: int, root: Hashable | None = None
) -> Paths:
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    if root is None:
        if len(chain.roots) != 1:
            raise ValueError("pass root explicitly when the chain has several roots")
        r = chain.roots[0]
    else:
        r = chain.index[root]
    cum, nxt, cost, absorb = _tables(chain)
    rng = np.random.default_rng(seed)
    state = np.full(n_paths, r, dtype=np.int64)
    paid = np.zeros(n_paths, dtype=np.int64)
    out = np.zeros(n_paths, dtype=np.int8)
    active = np.arange(n_paths)
    for _ in range(chain.n + 1):  # a DAG path visits each state at most once
        if active.size == 0:
            break
        s = state[active]
        u = rng.random(active.size)
        k = (cum[s] < u[:, None]).sum(axis=1)
        paid[active] += cost[s, k]
        ab = absorb[s, k]
        state[active] = nxt[s, k]
        done = ab > 0
        out[active[done]] = ab[done]
        active = active[~done]
    else:
        raise RuntimeError("simulation did not terminate; the chain should be a DAG")
    return Paths(out, paid)
```

- [ ] **Step 4: Implement compare.py**

```python
# src/gacha/mc/compare.py
"""Exact-vs-Monte-Carlo comparison table with standard errors."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import numpy as np
import pandas as pd

from gacha.kernel.types import HittingTime
from gacha.risk import metrics as rk

from .simulate import Paths


def compare(
    ht: HittingTime, paths: Paths, thresholds: Sequence[int] = (), tol_se: float = 3.0
) -> pd.DataFrame:
    n = len(paths.paid)
    rows: list[dict[str, float | str | bool]] = []

    def add(metric: str, exact: float, mc: float, se: float) -> None:
        z = (mc - exact) / se if se > 0 else 0.0
        rows.append({"metric": metric, "exact": exact, "mc": mc, "se": se, "z": z, "ok": abs(z) <= tol_se})

    p = ht.p_success
    add("p_success", p, float((paths.absorb == 1).mean()), sqrt(max(p * (1 - p), 1e-300) / n))
    add("mean", rk.mean(ht), float(paths.paid.mean()), float(paths.paid.std(ddof=1)) / sqrt(n))
    for b in thresholds:
        s = rk.survival(ht, int(b))
        add(f"survival_at_{b}", s, float((paths.paid >= b).mean()), sqrt(max(s * (1 - s), 1e-300) / n))
    return pd.DataFrame(rows)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_mc.py -v --durations=5`
Expected: all PASS; each parity test under 30 s.

- [ ] **Step 6: Commit**

```bash
git add src/gacha/mc tests/test_mc.py
git commit -m "feat(mc): vectorized Monte Carlo on the enumerated chain with parity checks

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Plot helpers, experiment infrastructure and CLI

**Files:**
- Create: `src/gacha/plots/style.py`, `src/gacha/plots/waiting_time.py`, `src/gacha/plots/heatmaps.py`, `src/gacha/experiments/_io.py`
- Modify: `src/gacha/experiments/__init__.py`, `src/gacha/cli.py`
- Test: `tests/test_plots_cli.py`

**Interfaces:**
- Produces: `PALETTE`, `INK`, `INK_2`, `MUTED`, `GRID`, `SEQUENTIAL_CMAP`, `apply_style()`, `new_figure(width, height, ncols=1, nrows=1) -> (fig, axes)`, `save(fig, out_dir, name) -> list[Path]`; `plot_pmf(ax, pmf, label=None, color=None)`, `plot_cdf(...)`, `plot_survival(...)`, `plot_schedule(ax, probs, label=None, color=None)`; `heatmap(ax, z, x, y, xlabel, ylabel, cbar_label, cmap=..., vmin=None, vmax=None)`; `ensure_dirs(out_dir) -> (tables_dir, figures_dir)`, `write_table(df, out_dir, name) -> list[Path]`, `update_results_md(path, section, lines)`; `EXPERIMENTS: list[str]`, `run_experiment(name, out_dir, results_md=None) -> dict`; `main(argv=None) -> int` with subcommands `evaluate` and `experiment`.
- The palette is the dataviz reference categorical palette, validated in light mode (all checks pass; slots 3-5 are below 3:1 contrast on white, which is why every multi-series figure carries a legend and every experiment writes a table).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_plots_cli.py
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gacha.cli import main
from gacha.experiments import EXPERIMENTS, run_experiment
from gacha.experiments._io import ensure_dirs, update_results_md, write_table
from gacha.plots.heatmaps import heatmap
from gacha.plots.style import PALETTE, apply_style, new_figure, save
from gacha.plots.waiting_time import plot_cdf, plot_pmf, plot_schedule, plot_survival


def test_palette_and_figure_roundtrip(tmp_path: Path):
    assert len(PALETTE) == 8 and PALETTE[0] == "#2a78d6"
    apply_style()
    fig, ax = new_figure()
    pmf = np.array([0.0, 0.5, 0.3, 0.2])
    plot_pmf(ax, pmf, label="a")
    plot_cdf(ax, pmf, label="b")
    plot_survival(ax, pmf, label="c")
    plot_schedule(ax, [0.1, 0.5, 1.0], label="d")
    ax.legend()
    paths = save(fig, tmp_path, "fig")
    assert [p.name for p in paths] == ["fig.png", "fig.pdf"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_heatmap_draws(tmp_path: Path):
    apply_style()
    fig, ax = new_figure()
    z = np.arange(12, dtype=float).reshape(3, 4)
    im = heatmap(ax, z, x=np.array([0, 5, 10, 15]), y=np.array([1, 2, 3]), xlabel="x", ylabel="y", cbar_label="z")
    assert im.get_array().shape == (3, 4)
    save(fig, tmp_path, "hm")


def test_write_table_and_results_md(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2], "b": [0.123456, 2.5]})
    paths = write_table(df, tmp_path, "t")
    tables, figures = ensure_dirs(tmp_path)
    assert (tables / "t.csv").exists() and (tables / "t.md").exists()
    assert "| a | b |" in (tables / "t.md").read_text()
    md = tmp_path / "results.md"
    update_results_md(md, "e00", ["- x: 1"])
    update_results_md(md, "e01", ["- y: 2"])
    update_results_md(md, "e00", ["- x: 3"])
    text = md.read_text()
    assert "## e00\n\n- x: 3" in text and "- x: 1" not in text and "## e01\n\n- y: 2" in text


def test_registry_is_a_list_of_strings():
    assert isinstance(EXPERIMENTS, list)
    with pytest.raises(KeyError):
        run_experiment("does_not_exist", Path("."))


def test_cli_evaluate_prints_metrics(capsys):
    rc = main(["evaluate", "--pity", "70", "--banner-pulls", "50", "--budget", "30", "--realized", "60"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "P(success within 30 paid)" in out
    assert "P(T >= 60) from a fresh banner" in out


def test_cli_evaluate_already_owned(capsys):
    rc = main(["evaluate", "--copies", "1", "--up-obtained", "1", "--budget", "999"])
    out = capsys.readouterr().out
    assert rc == 0
    assert re.search(r"P\(success\)\s+1\.0000", out)
    assert re.search(r"P\(success within 999 paid\)\s+1\.0000", out)


def test_cli_evaluate_plan_file(tmp_path: Path, capsys):
    plan = tmp_path / "plan.toml"
    plan.write_text(
        'dossier0 = false\n[start]\nt = 10\n[[banners]]\ntarget_copies = 1\ncap = 120\n'
        '[[banners]]\ntarget_copies = 0\ncap = 0\n[[banners]]\ntarget_copies = 1\ncap = 120\n'
    )
    rc = main(["evaluate", "--plan", str(plan), "--budget", "150"])
    assert rc == 0
    assert "P(success within 150 paid)" in capsys.readouterr().out


def test_cli_unknown_experiment(capsys):
    assert main(["experiment", "nope"]) == 2
    assert "unknown experiment" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plots_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gacha.plots.style'`.

- [ ] **Step 3: Implement the plot modules**

`src/gacha/plots/style.py`:
```python
"""Figure style: validated categorical palette (dataviz reference, light mode) and chrome."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#ffffff"
SEQUENTIAL_CMAP = "Blues"  # one hue, light -> dark


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.prop_cycle": cycler(color=PALETTE),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )


def new_figure(width: float = 6.0, height: float = 3.6, ncols: int = 1, nrows: int = 1):
    return plt.subplots(nrows, ncols, figsize=(width, height), constrained_layout=True)


def save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths
```

`src/gacha/plots/waiting_time.py`:
```python
"""Line plots for PMF, CDF, survival and pity schedules (x axis = paid pulls)."""

from __future__ import annotations

import numpy as np


def plot_pmf(ax, pmf, label=None, color=None):
    pmf = np.asarray(pmf, dtype=float)
    ax.plot(np.arange(len(pmf)), pmf, drawstyle="steps-mid", label=label, color=color)
    ax.set_xlabel("paid pulls")
    ax.set_ylabel("probability")


def plot_cdf(ax, pmf, label=None, color=None):
    pmf = np.asarray(pmf, dtype=float)
    ax.plot(np.arange(len(pmf)), np.cumsum(pmf), drawstyle="steps-post", label=label, color=color)
    ax.set_xlabel("paid pulls b")
    ax.set_ylabel("P(T ≤ b)")


def plot_survival(ax, pmf, label=None, color=None):
    pmf = np.asarray(pmf, dtype=float)
    c = np.cumsum(pmf)
    surv = 1.0 - np.concatenate([[0.0], c[:-1]])  # P(T >= b)
    ax.plot(np.arange(len(pmf)), surv, drawstyle="steps-post", label=label, color=color)
    ax.set_xlabel("paid pulls b")
    ax.set_ylabel("P(T ≥ b)")


def plot_schedule(ax, probs, label=None, color=None):
    probs = np.asarray(probs, dtype=float)
    ax.plot(np.arange(len(probs)), probs, drawstyle="steps-post", label=label, color=color)
    ax.set_xlabel("pity counter t")
    ax.set_ylabel("P(6★ on the next pull)")
```

`src/gacha/plots/heatmaps.py`:
```python
"""Single-hue heatmaps over a regular (x, y) grid."""

from __future__ import annotations

import numpy as np

from .style import SEQUENTIAL_CMAP


def heatmap(ax, z, x, y, xlabel, ylabel, cbar_label, cmap=SEQUENTIAL_CMAP, vmin=None, vmax=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = (x[1] - x[0]) if len(x) > 1 else 1.0
    dy = (y[1] - y[0]) if len(y) > 1 else 1.0
    im = ax.imshow(
        np.asarray(z, dtype=float),
        origin="lower",
        aspect="auto",
        extent=(x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.grid(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.figure.colorbar(im, ax=ax).set_label(cbar_label)
    return im
```

- [ ] **Step 4: Implement the experiment infrastructure**

`src/gacha/experiments/_io.py`:
```python
"""Output helpers shared by experiments: tables (CSV + Markdown) and docs/results.md sections."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dirs(out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1e-3 or abs(v) >= 1e4 else f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(" --- " for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, out_dir: Path, name: str) -> list[Path]:
    tables, _ = ensure_dirs(out_dir)
    csv = tables / f"{name}.csv"
    md = tables / f"{name}.md"
    df.to_csv(csv, index=False)
    md.write_text(to_markdown(df))
    return [csv, md]


def update_results_md(path: Path, section: str, lines: list[str]) -> None:
    """Replace (or append) the ``## section`` block of a Markdown file."""
    path = Path(path)
    text = path.read_text() if path.exists() else "# Results\n"
    parts = text.split("\n## ")
    head, blocks = parts[0], parts[1:]
    kept = [b for b in blocks if b.split("\n", 1)[0].strip() != section]
    kept.append(f"{section}\n\n" + "\n".join(lines) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(head.rstrip("\n") + "\n\n## " + "\n## ".join(b.rstrip("\n") + "\n" for b in kept))
```

`src/gacha/experiments/__init__.py`:
```python
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
```

- [ ] **Step 5: Implement the CLI**

```python
# src/gacha/cli.py
"""Command-line entry points: ``gacha evaluate`` and ``gacha experiment``."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from gacha.experiments import EXPERIMENTS, run_experiment
from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.endfield import BannerState, SingleBannerModel
from gacha.models.plan import Plan
from gacha.risk import metrics as rk
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules


def _plan_from_toml(path: Path) -> tuple[Plan, Plan]:
    data = tomllib.loads(path.read_text())
    rules = EndfieldCharacterRules(**data.get("rules", {}))
    banners = [BannerSpec(rules, int(b["target_copies"]), int(b["cap"])) for b in data["banners"]]
    s = data.get("start", {})
    start = BannerState(int(s.get("t", 0)), int(s.get("n", 0)), int(s.get("c", 0)), int(s.get("u", 0)))
    current = Plan(banners, start=start, dossier0=bool(data.get("dossier0", False)))
    fresh = Plan(banners)
    return current, fresh


def _evaluate(args: argparse.Namespace) -> int:
    if args.plan:
        model, fresh = _plan_from_toml(Path(args.plan))
    else:
        spec = BannerSpec(EndfieldCharacterRules(), args.target, args.cap)
        model = SingleBannerModel(
            spec,
            start=BannerState(args.pity, args.banner_pulls, args.copies, args.up_obtained),
            dossier=bool(args.dossier),
        )
        fresh = SingleBannerModel(spec)
    ht = hitting_time(EnumeratedChain.from_model(model))
    budget = args.budget if args.budget is not None else ht.horizon
    lines = [
        f"P(success)                    {ht.p_success:.4f}",
        f"P(success within {budget} paid)    {rk.completion(ht, budget):.4f}",
        f"mean paid pulls               {rk.mean(ht):.2f}",
        f"sd                            {rk.sd(ht):.2f}",
    ]
    if ht.p_success > 0:
        for a in (0.5, 0.9, 0.95, 0.99):
            lines.append(f"q{int(a * 100):<3d} (given success)         {rk.quantile(ht, a, 'success')}")
        lines.append(f"CVaR90 (given success)        {rk.cvar(ht, 0.9, 'success'):.2f}")
    lines.append(f"expected excess over budget   {rk.expected_excess(ht, budget):.2f}")
    if args.realized is not None:
        fresh_ht = hitting_time(EnumeratedChain.from_model(fresh))
        lines.append(f"P(T >= {args.realized}) from a fresh banner  {rk.survival(fresh_ht, args.realized):.4f}")
    print("\n".join(lines))
    return 0


def _experiment(args: argparse.Namespace) -> int:
    names = EXPERIMENTS if args.name == "all" else [args.name]
    for name in names:
        if name not in EXPERIMENTS:
            print(f"unknown experiment {name!r}; known: {', '.join(EXPERIMENTS) or '(none)'}", file=sys.stderr)
            return 2
        results_md = Path(args.results_md) if args.results_md else None
        headline = run_experiment(name, Path(args.out), results_md=results_md)
        print(f"{name}: " + ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in headline.items()))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gacha", description="Exact gacha waiting-time models")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="evaluate a personal banner state or a plan file")
    ev.add_argument("--pity", type=int, default=0, help="pity counter t (0-79)")
    ev.add_argument("--banner-pulls", type=int, default=0, help="counted pulls already made on this banner")
    ev.add_argument("--copies", type=int, default=0, help="UP copies already obtained on this banner")
    ev.add_argument("--up-obtained", type=int, choices=(0, 1), default=0, help="1 if the 120 guarantee is already void")
    ev.add_argument("--dossier", type=int, choices=(0, 1), default=0, help="1 if this banner received the 60-pull dossier")
    ev.add_argument("--target", type=int, default=1, help="wanted UP copies")
    ev.add_argument("--cap", type=int, default=120, help="maximum paid pulls on this banner")
    ev.add_argument("--budget", type=int, default=None, help="paid pulls available")
    ev.add_argument("--realized", type=int, default=None, help="paid pulls you actually needed; prints how unlucky that was")
    ev.add_argument("--plan", type=str, default=None, help="TOML plan file (multi-banner)")
    ev.set_defaults(func=_evaluate)

    ex = sub.add_parser("experiment", help="run a registered experiment or 'all'")
    ex.add_argument("name")
    ex.add_argument("--out", default="results")
    ex.add_argument("--results-md", default="docs/results.md")
    ex.set_defaults(func=_experiment)

    args = parser.parse_args(argv)
    return args.func(args)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_plots_cli.py -v && uv run gacha evaluate --pity 40 --banner-pulls 20 --budget 60`
Expected: all PASS; the CLI prints the metrics block.

- [ ] **Step 7: Commit**

```bash
git add src/gacha/plots src/gacha/experiments src/gacha/cli.py tests/test_plots_cli.py
git commit -m "feat: plot helpers, experiment registry/io and CLI (evaluate, experiment)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: Experiments E1 (paper reproduction) and E2 (Endfield single banner)

**Files:**
- Create: `src/gacha/experiments/e01_reproduce_paper.py`, `src/gacha/experiments/e02_endfield_single.py`
- Modify: `src/gacha/experiments/__init__.py` (append names to `EXPERIMENTS`)
- Test: `tests/test_experiments.py`

**Interfaces:**
- Consumes: everything from Tasks 2-11.
- Produces: `run(out_dir: Path) -> dict[str, float]` in each module; tables under `out_dir/tables/e0X_*.csv|.md`; figures under `out_dir/figures/e0X_fig_*.png|.pdf`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_experiments.py
from pathlib import Path

import pytest

from gacha.experiments import EXPERIMENTS, run_experiment


def _files(out: Path, kind: str, prefix: str) -> list[str]:
    return sorted(p.name for p in (out / kind).glob(f"{prefix}*"))


def test_e01_reproduce_paper(tmp_path: Path):
    head = run_experiment("e01_reproduce_paper", tmp_path, results_md=tmp_path / "results.md")
    assert head["E0"] == pytest.approx(62.34, abs=0.01)
    assert head["T10_q90"] == 719
    assert head["featured_mean"] == pytest.approx(93.51, abs=0.01)
    assert "e01_baselines.csv" in _files(tmp_path, "tables", "e01")
    assert "e01_fig_t10_pmf.png" in _files(tmp_path, "figures", "e01")
    assert "## e01_reproduce_paper" in (tmp_path / "results.md").read_text()


def test_e02_endfield_single(tmp_path: Path):
    head = run_experiment("e02_endfield_single", tmp_path)
    assert head["full_support_max"] == 115
    assert 40 < head["full_mean"] < 90
    assert head["two_copies_p_success"] == pytest.approx(1.0, abs=1e-9)
    assert "e02_variants.md" in _files(tmp_path, "tables", "e02")
    assert "e02_fig_pmf.pdf" in _files(tmp_path, "figures", "e02")


def test_registry_lists_e01_and_e02():
    assert "e01_reproduce_paper" in EXPERIMENTS and "e02_endfield_single" in EXPERIMENTS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_experiments.py -v`
Expected: FAIL with `KeyError: "unknown experiment 'e01_reproduce_paper'..."`.

- [ ] **Step 3: Implement e01_reproduce_paper.py**

```python
# src/gacha/experiments/e01_reproduce_paper.py
"""E1: reproduce the figures and tables of Hou, Zhu & Zhang (2026) as an engine validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gacha.kernel.backward import state_values
from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.paper import Featured5050Model, SingleCounterModel
from gacha.plots.style import apply_style, new_figure, save
from gacha.plots.waiting_time import plot_cdf, plot_pmf, plot_schedule, plot_survival
from gacha.risk import metrics as rk
from gacha.rules.paper import PaperSchedule

from ._io import ensure_dirs, write_table

ALPHAS = (0.5, 0.75, 0.9, 0.95, 0.99)
BUDGETS = (600, 650, 700, 719, 741, 775, 800)


def run(out_dir: Path) -> dict[str, float]:
    apply_style()
    _, figures = ensure_dirs(out_dir)
    sched = PaperSchedule()
    hard = sched.hard_only()
    soft_chain = EnumeratedChain.from_model(SingleCounterModel(sched.probs))
    soft = hitting_time(soft_chain)
    hard_ht = hitting_time(EnumeratedChain.from_model(SingleCounterModel(hard.probs)))
    t10 = rk.from_pmf(rk.convolve(soft.pmf_stop, 10))
    feat = hitting_time(EnumeratedChain.from_model(Featured5050Model(sched.probs)))
    sv = state_values(soft_chain)

    write_table(
        pd.DataFrame(
            {
                "model": ["uncapped geometric", "hard pity only (90)", "soft-pity schedule (1)"],
                "expected_draws": [1.0 / sched.base, rk.mean(hard_ht), rk.mean(soft)],
            }
        ),
        out_dir,
        "e01_baselines",
    )
    ts = [0, 70, 72, 73, 80, 89]
    write_table(
        pd.DataFrame({"pity_t": ts, "expected_remaining": [sv.expectation[soft_chain.index[(t,)]] for t in ts]}),
        out_dir,
        "e01_expected_remaining",
    )
    write_table(
        pd.DataFrame(
            {
                "alpha": ALPHAS,
                "VaR": [rk.quantile(t10, a) for a in ALPHAS],
                "CVaR": [rk.cvar(t10, a) for a in ALPHAS],
                "tail_prob_at_VaR": [rk.survival(t10, rk.quantile(t10, a)) for a in ALPHAS],
            }
        ),
        out_dir,
        "e01_t10_var_cvar",
    )
    write_table(
        pd.DataFrame(
            {
                "budget": BUDGETS,
                "completion": [rk.completion(t10, b) for b in BUDGETS],
                "expected_excess": [rk.expected_excess(t10, b) for b in BUDGETS],
            }
        ),
        out_dir,
        "e01_t10_budget",
    )

    fig, ax = new_figure()
    plot_schedule(ax, hard.probs, label="hard pity only")
    plot_schedule(ax, sched.probs, label="soft-pity schedule")
    ax.set_title("Paper schedule: state-dependent success probability")
    ax.legend()
    save(fig, figures, "e01_fig_schedule")

    fig, ax = new_figure()
    plot_pmf(ax, soft.pmf_stop)
    ax.set_title("Single-stage waiting-time PMF (paper Fig. 6)")
    save(fig, figures, "e01_fig_single_pmf")

    fig, ax = new_figure()
    plot_cdf(ax, soft.pmf_stop, label="CDF")
    plot_survival(ax, soft.pmf_stop, label="survival P(T ≥ b)")
    ax.set_ylabel("probability")
    ax.legend()
    save(fig, figures, "e01_fig_single_cdf")

    fig, ax = new_figure()
    plot_pmf(ax, t10.pmf_stop)
    ax.set_xlim(400, 900)
    ax.set_title("Ten independent rare items (paper Fig. 8)")
    save(fig, figures, "e01_fig_t10_pmf")

    levels = np.linspace(0.5, 0.999, 200)
    fig, ax = new_figure()
    ax.plot(levels, [rk.quantile(t10, a) for a in levels])
    ax.set_xlabel("quantile level")
    ax.set_ylabel("paid pulls")
    ax.set_title("Exact quantile curve for T10 (paper Fig. 9)")
    save(fig, figures, "e01_fig_t10_quantiles")

    fig, ax = new_figure()
    plot_survival(ax, t10.pmf_stop)
    ax.set_xlim(500, 850)
    ax.set_title("Right-tail P(T10 ≥ b) (paper Fig. 10)")
    save(fig, figures, "e01_fig_t10_tail")

    levels = np.linspace(0.7, 0.995, 60)
    fig, ax = new_figure()
    ax.plot(levels, [rk.quantile(t10, a) for a in levels], label="VaR")
    ax.plot(levels, [rk.cvar(t10, a) for a in levels], label="CVaR (tail conditional mean)")
    ax.set_xlabel("risk level α")
    ax.set_ylabel("paid pulls")
    ax.legend()
    save(fig, figures, "e01_fig_t10_var_cvar")

    fig, ax = new_figure()
    plot_pmf(ax, soft.pmf_stop, label="any rare item")
    plot_pmf(ax, feat.pmf_stop, label="featured target, no guarantee")
    ax.legend()
    ax.set_title("Effect of a 50/50 guarantee state (paper Fig. 19)")
    save(fig, figures, "e01_fig_featured_pmf")

    return {
        "E0": rk.mean(soft),
        "V0": rk.var(soft),
        "hard_only_E0": rk.mean(hard_ht),
        "T10_mean": rk.mean(t10),
        "T10_q90": rk.quantile(t10, 0.9),
        "featured_mean": rk.mean(feat),
    }
```

- [ ] **Step 4: Implement e02_endfield_single.py**

```python
# src/gacha/experiments/e02_endfield_single.py
"""E2: paid pulls to the first UP on one Endfield banner, rule variants overlaid."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.endfield import SingleBannerModel
from gacha.plots.style import apply_style, new_figure, save
from gacha.plots.waiting_time import plot_cdf, plot_pmf
from gacha.risk import metrics as rk
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules

from ._io import ensure_dirs, write_table


def _ht(model):
    return hitting_time(EnumeratedChain.from_model(model))


def run(out_dir: Path) -> dict[str, float]:
    apply_style()
    _, figures = ensure_dirs(out_dir)
    bare = EndfieldCharacterRules(
        guarantee_pull=None, vacuum_at=None, dossier_at=None, potential_every=None, free_start_pulls=0
    )
    full = EndfieldCharacterRules()
    variants = [
        ("80 pity + 50/50 only (cap 480)", SingleBannerModel(BannerSpec(bare, 1, 480))),
        ("+ 120 guarantee", SingleBannerModel(BannerSpec(replace(bare, guarantee_pull=120), 1, 120))),
        ("full rules, no dossier (f=5)", SingleBannerModel(BannerSpec(full, 1, 120))),
        ("full rules, with dossier (f=15)", SingleBannerModel(BannerSpec(full, 1, 120), dossier=True)),
    ]
    rows, hts = [], []
    for name, model in variants:
        ht = _ht(model)
        hts.append((name, ht))
        rows.append(
            {
                "variant": name,
                "p_success": ht.p_success,
                "mean": rk.mean(ht),
                "sd": rk.sd(ht),
                "q50": rk.quantile(ht, 0.5),
                "q90": rk.quantile(ht, 0.9),
                "q95": rk.quantile(ht, 0.95),
                "q99": rk.quantile(ht, 0.99),
                "support_max": int(np.nonzero(ht.pmf_stop)[0].max()),
            }
        )
    write_table(pd.DataFrame(rows), out_dir, "e02_variants")

    two = _ht(SingleBannerModel(BannerSpec(full, 2, 300)))
    write_table(
        pd.DataFrame(
            [{"target_copies": 2, "cap": 300, "p_success": two.p_success, "mean": rk.mean(two),
              "q90": rk.quantile(two, 0.9), "q99": rk.quantile(two, 0.99)}]
        ),
        out_dir,
        "e02_two_copies",
    )

    fig, ax = new_figure(6.5, 3.8)
    for name, ht in hts:
        plot_pmf(ax, ht.pmf_stop, label=name)
    ax.set_xlim(0, 200)
    ax.set_title("Paid pulls to the first UP: Endfield rule variants")
    ax.legend()
    save(fig, figures, "e02_fig_pmf")

    fig, ax = new_figure(6.5, 3.8)
    for name, ht in hts:
        plot_cdf(ax, ht.pmf_stop, label=name)
    ax.set_xlim(0, 200)
    ax.legend()
    save(fig, figures, "e02_fig_cdf")

    fig, ax = new_figure()
    plot_pmf(ax, two.pmf_stop)
    ax.set_title("Two UP copies on one banner (240-pull potential included)")
    save(fig, figures, "e02_fig_two_copies_pmf")

    full_ht = hts[2][1]
    return {
        "full_mean": rk.mean(full_ht),
        "full_sd": rk.sd(full_ht),
        "full_q90": rk.quantile(full_ht, 0.9),
        "full_support_max": int(np.nonzero(full_ht.pmf_stop)[0].max()),
        "two_copies_mean": rk.mean(two),
        "two_copies_p_success": two.p_success,
    }
```

- [ ] **Step 5: Register both experiments**

In `src/gacha/experiments/__init__.py` set:
```python
EXPERIMENTS: list[str] = ["e01_reproduce_paper", "e02_endfield_single"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_experiments.py -v && uv run gacha experiment e01_reproduce_paper --out results && ls results/figures | head`
Expected: tests PASS; PNG/PDF files listed. Open `results/figures/e02_fig_pmf.png` and check: four labeled series, a visible spike at 115 for the full-rules variant, legend present, no clipped labels.

- [ ] **Step 7: Commit**

```bash
git add src/gacha/experiments tests/test_experiments.py
git commit -m "feat(experiments): E1 paper reproduction and E2 Endfield single-banner variants

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: Experiments E3 (personal state) and E4 (carry-over)

**Files:**
- Create: `src/gacha/experiments/e03_personal_state.py`, `src/gacha/experiments/e04_carry_over.py`
- Modify: `src/gacha/experiments/__init__.py`, `tests/test_experiments.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_experiments.py`:
```python
def test_e03_personal_state(tmp_path: Path):
    head = run_experiment("e03_personal_state", tmp_path)
    assert head["E_t79_n0"] < head["E_fresh"]
    assert head["E_t0_n100"] < head["E_fresh"]
    assert head["monotone_in_t0"] == 1.0
    assert "e03_fig_expected_remaining.png" in _files(tmp_path, "figures", "e03")
    assert "e03_value_of_pity.csv" in _files(tmp_path, "tables", "e03")


def test_e04_carry_over(tmp_path: Path):
    head = run_experiment("e04_carry_over", tmp_path)
    assert head["gap_K2"] > 1e-3 and head["gap_K3"] > 1e-3
    assert head["exact_mean_K2"] < head["iid_mean_K2"]  # entering pity and dossier help
    assert "e04_carry_over.md" in _files(tmp_path, "tables", "e04")
    assert "e04_fig_cdf.png" in _files(tmp_path, "figures", "e04")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_experiments.py -v -k "e03 or e04"`
Expected: FAIL with `KeyError`.

- [ ] **Step 3: Implement e03_personal_state.py**

```python
# src/gacha/experiments/e03_personal_state.py
"""E3: value of the current state (t, n): expected remaining paid pulls and success within b."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gacha.kernel.backward import state_values, success_within
from gacha.kernel.chain import EnumeratedChain
from gacha.models.endfield import BannerState, SingleBannerModel
from gacha.plots.heatmaps import heatmap
from gacha.plots.style import apply_style, new_figure, save
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules

from ._io import ensure_dirs, write_table

BUDGETS = (30, 60, 90)


def run(out_dir: Path) -> dict[str, float]:
    apply_style()
    _, figures = ensure_dirs(out_dir)
    rules = EndfieldCharacterRules(free_start_pulls=0)  # n == paid pulls already made
    spec = BannerSpec(rules, 1, 120)
    nt, nn = rules.hard_pity, 120
    roots = [BannerState(t, n, 0, 0) for t in range(nt) for n in range(nn)]
    chain = EnumeratedChain.from_model(SingleBannerModel(spec), roots=roots)
    sv = state_values(chain)
    s = success_within(chain, max(BUDGETS))

    def grid(values: np.ndarray) -> np.ndarray:
        return np.array([[values[chain.index[BannerState(t, n, 0, 0)]] for n in range(nn)] for t in range(nt)])

    e_grid = grid(sv.expectation)
    p_grids = {b: grid(s[b]) for b in BUDGETS}

    fig, ax = new_figure(6.5, 4.2)
    heatmap(ax, e_grid, x=np.arange(nn), y=np.arange(nt), xlabel="pulls already made on this banner n",
            ylabel="pity counter t", cbar_label="expected remaining paid pulls to first UP")
    ax.set_title("Expected remaining paid pulls (u = 0, no copies yet)")
    save(fig, figures, "e03_fig_expected_remaining")

    fig, axes = new_figure(11, 3.6, ncols=3)
    for ax, b in zip(axes, BUDGETS, strict=True):
        heatmap(ax, p_grids[b], x=np.arange(nn), y=np.arange(nt), xlabel="pulls already made n",
                ylabel="pity counter t", cbar_label=f"P(first UP within {b} paid pulls)", vmin=0, vmax=1)
    save(fig, figures, "e03_fig_success_within")

    full = EndfieldCharacterRules()
    roots5 = [BannerState(t, 0, 0, 0) for t in range(nt)]
    chain5 = EnumeratedChain.from_model(SingleBannerModel(BannerSpec(full, 1, 120)), roots=roots5)
    sv5 = state_values(chain5)
    e_t0 = [float(sv5.expectation[chain5.index[r]]) for r in roots5]
    write_table(pd.DataFrame({"t0": range(nt), "expected_paid_pulls": e_t0}), out_dir, "e03_value_of_pity")
    fig, ax = new_figure()
    ax.plot(range(nt), e_t0)
    ax.set_xlabel("entering pity t0")
    ax.set_ylabel("expected paid pulls to first UP")
    ax.set_title("The value of pity (default rules, f = 5)")
    save(fig, figures, "e03_fig_value_of_pity")

    selected = [(0, 0), (0, 60), (0, 100), (40, 40), (64, 0), (70, 0), (79, 0), (0, 119)]
    write_table(
        pd.DataFrame(
            [{"t": t, "n": n, "expected_remaining": e_grid[t, n], "p_within_30": p_grids[30][t, n],
              "p_within_60": p_grids[60][t, n], "p_within_90": p_grids[90][t, n]} for t, n in selected]
        ),
        out_dir,
        "e03_selected_states",
    )
    return {
        "E_fresh": float(e_grid[0, 0]),
        "E_t79_n0": float(e_grid[79, 0]),
        "E_t0_n100": float(e_grid[0, 100]),
        "monotone_in_t0": float(bool(np.all(np.diff(e_t0) <= 1e-9))),
    }
```

- [ ] **Step 4: Implement e04_carry_over.py**

```python
# src/gacha/experiments/e04_carry_over.py
"""E4: exact multi-banner distribution (pity + dossier carried) vs the paper's i.i.d. convolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.endfield import SingleBannerModel
from gacha.models.plan import Plan
from gacha.plots.style import PALETTE, apply_style, new_figure, save
from gacha.risk import metrics as rk
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules

from ._io import ensure_dirs, write_table

KS = (2, 3)


def _padded_cdfs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = max(len(a), len(b))
    return np.cumsum(np.pad(a, (0, n - len(a)))), np.cumsum(np.pad(b, (0, n - len(b))))


def run(out_dir: Path) -> dict[str, float]:
    apply_style()
    _, figures = ensure_dirs(out_dir)
    spec = BannerSpec(EndfieldCharacterRules(), 1, 120)
    single = hitting_time(EnumeratedChain.from_model(SingleBannerModel(spec)))
    rows, curves = [], {}
    for k in KS:
        exact = hitting_time(EnumeratedChain.from_model(Plan([spec] * k)))
        iid = rk.from_pmf(rk.convolve(single.pmf_stop, k))
        ce, cc = _padded_cdfs(exact.pmf_stop, iid.pmf_stop)
        curves[k] = (ce, cc)
        rows.append(
            {
                "banners": k,
                "exact_mean": rk.mean(exact), "iid_mean": rk.mean(iid),
                "exact_q50": rk.quantile(exact, 0.5), "iid_q50": rk.quantile(iid, 0.5),
                "exact_q90": rk.quantile(exact, 0.9), "iid_q90": rk.quantile(iid, 0.9),
                "exact_q95": rk.quantile(exact, 0.95), "iid_q95": rk.quantile(iid, 0.95),
                "max_cdf_gap": float(np.abs(ce - cc).max()),
                "exact_p_within_100_per_banner": rk.completion(exact, 100 * k),
                "iid_p_within_100_per_banner": rk.completion(iid, 100 * k),
            }
        )
    df = pd.DataFrame(rows)
    write_table(df, out_dir, "e04_carry_over")

    fig, axes = new_figure(10, 3.6, ncols=len(KS))
    for ax, k in zip(axes, KS, strict=True):
        ce, cc = curves[k]
        ax.plot(ce, label="exact (pity + dossier carried)", color=PALETTE[0])
        ax.plot(cc, label="i.i.d. convolution (paper Eq. 2)", color=PALETTE[1], linestyle="--")
        ax.set_title(f"{k} consecutive wanted banners")
        ax.set_xlabel("paid pulls b")
        ax.set_ylabel("P(all UPs, T ≤ b)")
    axes[0].legend()
    save(fig, figures, "e04_fig_cdf")

    fig, ax = new_figure()
    for k in KS:
        ce, cc = curves[k]
        ax.plot(ce - cc, label=f"{k} banners")
    ax.set_xlabel("paid pulls b")
    ax.set_ylabel("exact CDF − i.i.d. CDF")
    ax.legend()
    save(fig, figures, "e04_fig_gap")

    by_k = {int(r["banners"]): r for r in rows}
    return {
        "gap_K2": by_k[2]["max_cdf_gap"], "gap_K3": by_k[3]["max_cdf_gap"],
        "exact_mean_K2": by_k[2]["exact_mean"], "iid_mean_K2": by_k[2]["iid_mean"],
        "exact_q90_K3": by_k[3]["exact_q90"], "iid_q90_K3": by_k[3]["iid_q90"],
    }
```

- [ ] **Step 5: Register, run, commit**

Append `"e03_personal_state", "e04_carry_over"` to `EXPERIMENTS`.

Run: `uv run pytest tests/test_experiments.py -v && uv run gacha experiment e04_carry_over --out results`
Expected: PASS; look at `results/figures/e04_fig_cdf.png`: two panels, solid vs dashed curves, legend in the first panel.

```bash
git add src/gacha/experiments tests/test_experiments.py
git commit -m "feat(experiments): E3 personal-state heatmaps and E4 cross-banner carry-over gap

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 14: Experiments E5 (tail risk), E6 (Monte Carlo convergence), E7 (sensitivity)

**Files:**
- Create: `src/gacha/experiments/e05_tail_risk.py`, `src/gacha/experiments/e06_mc_convergence.py`, `src/gacha/experiments/e07_sensitivity.py`
- Modify: `src/gacha/experiments/__init__.py`, `tests/test_experiments.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_experiments.py`:
```python
def test_e05_tail_risk(tmp_path: Path):
    head = run_experiment("e05_tail_risk", tmp_path)
    assert head["q90_K1"] < head["q90_K3"] < head["q90_K5"]
    assert 0.0 < head["completion_60_per_banner_K5"] < 1.0
    assert "e05_tail_risk.csv" in _files(tmp_path, "tables", "e05")
    assert "e05_fig_completion.png" in _files(tmp_path, "figures", "e05")


def test_e06_mc_convergence(tmp_path: Path):
    head = run_experiment("e06_mc_convergence", tmp_path)
    assert abs(head["mc_200k"] - head["exact_survival_90"]) < 0.005
    assert head["max_abs_z"] < 3.5
    assert "e06_convergence.md" in _files(tmp_path, "tables", "e06")
    assert "e06_fig_convergence.png" in _files(tmp_path, "figures", "e06")


def test_e07_sensitivity(tmp_path: Path):
    head = run_experiment("e07_sensitivity", tmp_path)
    assert head["mean_up_share_0.7"] < head["mean_up_share_0.5"]
    assert head["mean_guarantee_100"] < head["mean_guarantee_140"]
    assert "e07_sensitivity.md" in _files(tmp_path, "tables", "e07")
    assert "e07_fig_sensitivity.png" in _files(tmp_path, "figures", "e07")


def test_registry_complete():
    assert EXPERIMENTS == [
        "e01_reproduce_paper", "e02_endfield_single", "e03_personal_state", "e04_carry_over",
        "e05_tail_risk", "e06_mc_convergence", "e07_sensitivity",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_experiments.py -v -k "e05 or e06 or e07 or complete"`
Expected: FAIL with `KeyError` / registry mismatch.

- [ ] **Step 3: Implement e05_tail_risk.py**

```python
# src/gacha/experiments/e05_tail_risk.py
"""E5: tail-risk summaries for 1-5 consecutive wanted banners and a completion heatmap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.plan import Plan
from gacha.plots.heatmaps import heatmap
from gacha.plots.style import apply_style, new_figure, save
from gacha.risk import metrics as rk
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules

from ._io import ensure_dirs, write_table

KS = (1, 2, 3, 4, 5)
ALPHAS = (0.5, 0.75, 0.9, 0.95, 0.99)


def run(out_dir: Path) -> dict[str, float]:
    apply_style()
    _, figures = ensure_dirs(out_dir)
    spec = BannerSpec(EndfieldCharacterRules(), 1, 120)
    budgets = np.arange(0, 5 * 115 + 1, 5)
    rows, completion = [], []
    hts = {}
    for k in KS:
        ht = hitting_time(EnumeratedChain.from_model(Plan([spec] * k)))
        hts[k] = ht
        row = {"banners": k, "mean": rk.mean(ht), "sd": rk.sd(ht)}
        row.update({f"q{round(a * 100)}": rk.quantile(ht, a) for a in ALPHAS})
        row.update({"cvar90": rk.cvar(ht, 0.9), "cvar95": rk.cvar(ht, 0.95)})
        for per in (60, 90):
            row[f"excess_at_{per}_per_banner"] = rk.expected_excess(ht, per * k)
            row[f"completion_{per}_per_banner"] = rk.completion(ht, per * k)
        rows.append(row)
        completion.append([rk.completion(ht, int(b)) for b in budgets])
    write_table(pd.DataFrame(rows), out_dir, "e05_tail_risk")

    fig, ax = new_figure(6.5, 3.6)
    heatmap(ax, np.array(completion), x=budgets, y=np.array(KS), xlabel="budget in paid pulls",
            ylabel="consecutive wanted banners", cbar_label="P(all UPs within budget)", vmin=0, vmax=1)
    ax.set_yticks(list(KS))
    save(fig, figures, "e05_fig_completion")

    levels = np.linspace(0.7, 0.995, 60)
    fig, ax = new_figure()
    ax.plot(levels, [rk.quantile(hts[3], a) for a in levels], label="VaR")
    ax.plot(levels, [rk.cvar(hts[3], a) for a in levels], label="CVaR")
    ax.set_xlabel("risk level α")
    ax.set_ylabel("paid pulls")
    ax.set_title("Three consecutive wanted banners")
    ax.legend()
    save(fig, figures, "e05_fig_var_cvar_K3")

    by_k = {int(r["banners"]): r for r in rows}
    return {
        "q90_K1": by_k[1]["q90"], "q90_K3": by_k[3]["q90"], "q90_K5": by_k[5]["q90"],
        "mean_K5": by_k[5]["mean"], "cvar95_K5": by_k[5]["cvar95"],
        "completion_60_per_banner_K5": by_k[5]["completion_60_per_banner"],
    }
```

- [ ] **Step 4: Implement e06_mc_convergence.py**

```python
# src/gacha/experiments/e06_mc_convergence.py
"""E6: Monte Carlo estimates converge to the exact tail probability (paper Fig. 11 style)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.mc.compare import compare
from gacha.mc.simulate import simulate
from gacha.models.endfield import SingleBannerModel
from gacha.plots.style import PALETTE, apply_style, new_figure, save
from gacha.risk import metrics as rk
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules

from ._io import ensure_dirs, write_table

THRESHOLD = 90
NS = (100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 200_000)
SEED = 20260923


def run(out_dir: Path) -> dict[str, float]:
    apply_style()
    _, figures = ensure_dirs(out_dir)
    chain = EnumeratedChain.from_model(SingleBannerModel(BannerSpec(EndfieldCharacterRules(), 1, 120)))
    ht = hitting_time(chain)
    exact = rk.survival(ht, THRESHOLD)
    paths = simulate(chain, NS[-1], seed=SEED)
    hits = np.cumsum((paths.paid >= THRESHOLD).astype(float))
    estimates = [float(hits[n - 1] / n) for n in NS]
    write_table(
        pd.DataFrame({"replications": NS, "mc_estimate": estimates, "exact": exact,
                      "abs_error": [abs(e - exact) for e in estimates]}),
        out_dir,
        "e06_convergence",
    )
    parity = compare(ht, paths, thresholds=(60, 90, 110))
    write_table(parity, out_dir, "e06_parity")

    fig, ax = new_figure()
    ax.plot(NS, estimates, marker="o", label="Monte Carlo estimate", color=PALETTE[0])
    ax.axhline(exact, label="exact (forward recursion)", color=PALETTE[1])
    ax.set_xscale("log")
    ax.set_xlabel("replications")
    ax.set_ylabel(f"P(T ≥ {THRESHOLD})")
    ax.legend()
    save(fig, figures, "e06_fig_convergence")
    return {
        "exact_survival_90": exact,
        "mc_200k": estimates[-1],
        "max_abs_z": float(parity["z"].abs().max()),
    }
```

- [ ] **Step 5: Implement e07_sensitivity.py**

```python
# src/gacha/experiments/e07_sensitivity.py
"""E7: sensitivity of the first-UP distribution to UP share, soft-pity start and the guarantee."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.endfield import SingleBannerModel
from gacha.plots.style import apply_style, new_figure, save
from gacha.risk import metrics as rk
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules

from ._io import ensure_dirs, write_table

GROUPS = ("up_share", "soft_pity_start", "guarantee_pull")


def run(out_dir: Path) -> dict[str, float]:
    apply_style()
    _, figures = ensure_dirs(out_dir)
    base = EndfieldCharacterRules()
    rows = []

    def evaluate(group: str, label: str, rules: EndfieldCharacterRules, cap: int) -> None:
        ht = hitting_time(EnumeratedChain.from_model(SingleBannerModel(BannerSpec(rules, 1, cap))))
        rows.append({"group": group, "setting": label, "mean": rk.mean(ht), "q90": rk.quantile(ht, 0.9),
                     "q99": rk.quantile(ht, 0.99), "p_success": ht.p_success})

    for q in (0.5, 0.6, 0.7):
        evaluate("up_share", f"up_share={q}", replace(base, up_share=q), 120)
    for s in (60, 65, 70):
        evaluate("soft_pity_start", f"soft_pity_start={s}", replace(base, soft_pity_start=s), 120)
    for g in (100, 120, 140, None):
        evaluate("guarantee_pull", f"guarantee={g}", replace(base, guarantee_pull=g), g if g else 240)
    df = pd.DataFrame(rows)
    write_table(df, out_dir, "e07_sensitivity")

    fig, axes = new_figure(11, 3.4, ncols=3)
    for ax, group in zip(axes, GROUPS, strict=True):
        sub = df[df["group"] == group]
        x = list(range(len(sub)))
        ax.plot(x, sub["mean"], marker="o", label="mean")
        ax.plot(x, sub["q90"], marker="s", label="90% quantile")
        ax.set_xticks(x, sub["setting"], rotation=20)
        ax.set_title(group)
    axes[0].set_ylabel("paid pulls")
    axes[0].legend()
    save(fig, figures, "e07_fig_sensitivity")

    by = {r["setting"]: r for r in rows}
    return {
        "mean_up_share_0.5": by["up_share=0.5"]["mean"],
        "mean_up_share_0.7": by["up_share=0.7"]["mean"],
        "mean_soft_60": by["soft_pity_start=60"]["mean"],
        "mean_guarantee_100": by["guarantee=100"]["mean"],
        "mean_guarantee_140": by["guarantee=140"]["mean"],
        "mean_guarantee_none": by["guarantee=None"]["mean"],
    }
```

- [ ] **Step 6: Register, run, commit**

Set `EXPERIMENTS` to the full seven-name list in order (as in `test_registry_complete`).

Run: `uv run pytest tests/test_experiments.py -v --durations=5`
Expected: all PASS.

```bash
git add src/gacha/experiments tests/test_experiments.py
git commit -m "feat(experiments): E5 tail risk, E6 Monte Carlo convergence, E7 sensitivity

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 15: Documentation, CLAUDE.md, project skills, full run

**Files:**
- Create: `docs/assumptions.md`, `docs/research-notes.md`, `CLAUDE.md`, `.claude/skills/add-mechanic/SKILL.md`, `.claude/skills/run-experiment/SKILL.md`
- Modify: `docs/results.md` (generated by running all experiments)

**Interfaces:** none (documentation). Before writing the two `SKILL.md` files, invoke `superpowers:writing-skills` and follow its format checks; the content below is the required substance.

- [ ] **Step 1: Write docs/assumptions.md**

```markdown
# Rule and policy assumptions register

Status: **confirmed** = user statement and endfield.wiki.gg agree; **assumed** = modeling choice to verify.
Every row maps to a parameter of `EndfieldCharacterRules` (`src/gacha/rules/endfield.py`) or to a
policy in `src/gacha/models/endfield.py` / `plan.py`. Change a rule here first, then in code (see the
`add-mechanic` skill).

| ID | Rule | Status | Parameter / code |
|----|------|--------|------------------|
| R1 | 500 合成玉 per pull; 1 源石 = 75 合成玉 | confirmed | Phase 2 cost layer |
| R2 | p_t = 0.008 (t ≤ 64); 0.008 + 0.05·(t−64) (65 ≤ t ≤ 78); p_79 = 1 | confirmed | base_rate, soft_pity_start, soft_pity_step, hard_pity |
| R3 | any 6★ from a counted pull resets t; t carries across limited banners | confirmed | `pull()` reset, `Plan` boundary |
| R4 | each counted 6★ is the UP w.p. 0.5, independently | confirmed | up_share |
| R5 | 120th counted pull of a banner is the UP if none obtained yet; one-time; resets t | confirmed (reset assumed) | guarantee_pull |
| R6 | at n = 30: 10 vacuum pulls, flat 0.008, no effect on t/n/u, copies add to c | isolation confirmed; rate assumed | vacuum_at, vacuum_pulls, vacuum_rate |
| R7 | n ≥ 60 at banner end → 10 free counted pulls at the start of the next banner | confirmed; counted in n assumed | dossier_at, dossier_pulls |
| R8 | every 240 counted pulls: +1 UP copy | confirmed | potential_every |
| R9 | 5 free banner-bound pulls at banner start | user; banner-bound assumed | free_start_pulls |
| R10 | off-rate 6★ composition (standard vs previous limited) | deferred (Phase 3) | — |
| R11 | 5★ layer (8%, one per 10 pulls) | deferred (Phase 3) | — |
| R12 | limited operator stays 3 banners | deferred (affects R10) | — |

| ID | Policy assumption | Code |
|----|-------------------|------|
| P1 | free pulls are used before paid pulls; paid = max(0, n − f) | `pull()` cost rule |
| P2 | free pulls are always used, even on skipped banners | `Plan` (cap 0 banners) |
| P3 | pull until target or cap; finish remaining free pulls after success | `status()` |
| P4 | T counts paid pulls only | `Transition.cost` |

## Open questions (spec §9)

1. Are the 5 free pulls banner-bound or generic permits (R9)?
2. Vacuum pull rate flat 0.008 (R6)?
3. Do dossier and free-start pulls count toward n (R7, R9)?
4. Does the guaranteed 120th pull reset t (R5)?
5. Weapon banner milestones at 40/100 and off-rate composition (Phase 3).
```

- [ ] **Step 2: Write docs/research-notes.md**

```markdown
# Research notes

Reference: Hou, Zhu & Zhang, *State-Dependent Asymmetry in Soft-Pity Gacha Waiting-Time Models*,
Symmetry 2026, 18(6), 1051, doi:10.3390/sym18061051 (CC BY 4.0).

## Contributions relative to the paper

1. **Two counters and an expiring guarantee.** Endfield's state is (t, n, c, u): the 80-pull pity t
   carries across banners, the banner-local count n drives 30/60/120/240 milestones, and the
   one-time 120 guarantee is void once u = 1. This asymmetry differs from the paper's (t, g)
   50/50-with-carry-over model; the guarantee truncates support inside a banner (paid ≤ 120 − f)
   but not across banners.
2. **Weighted hitting times on a DAG.** Free pulls are cost-0 transitions, so T is a
   cost-weighted absorption time. The backward recurrences (`kernel/backward.py`) generalize the
   paper's Propositions 1–2; the forward level-closure algorithm (`kernel/forward.py`) gives the
   exact PMF over paid pulls.
3. **Cross-banner coupling.** Entering pity and the dossier make consecutive banners dependent, so
   the paper's Eq. (2) i.i.d. convolution does not apply. E4 quantifies the CDF gap.
4. **Budget-oriented risk metrics** on defective distributions (P(success) < 1): completion
   probability, VaR/CVaR, expected excess, with explicit T_stop vs success-conditional conventions.

## Propositions to derive

- **Monotone value of pity.** E[T | t0] is non-increasing in t0. Sketch: n + E(0, n) is
  non-decreasing in n because delaying the start by one pull brings the guarantee one pull closer,
  so a chain from (0, n) can be coupled to finish at most one pull later than one from (0, n+1);
  then compare (t, n) with (t+1, n) pull by pull: a 6★ that happens only for the t+1 chain either
  ends it (probability q) or resets it to (0, n+k), and 0.5·E(0, n') ≤ E(t', n') + 0.5 gives the
  inequality. Checked numerically by `test_expected_pulls_monotone_in_entering_pity` and E3.
- **Stochastic ordering in the guarantee.** A smaller `guarantee_pull` gives a smaller T in the
  usual stochastic order (same coupling as the paper's Proposition 4 with a state-dependent hazard).
- **Vacuum bonus as an independent mixture.** The 30-pull vacuum adds a Binomial(10, q·0.008)
  number of copies independent of the chain; for the first-copy objective it multiplies the
  surviving mass at n = 30 by (1 − 0.996^10).
- **Support and spike.** With f free pulls, P(T = 120 − f) equals the probability of reaching pull
  120 without a UP; E2 reports it.

## Findings

(Experiments and tests append here: reproduction discrepancies, monotonicity checks, headline gaps.)
```

- [ ] **Step 3: Write CLAUDE.md**

```markdown
# gacha — exact gacha waiting-time models

Research codebase extending Hou, Zhu & Zhang (Symmetry 2026) to Arknights: Endfield. Design:
`docs/superpowers/specs/2026-09-16-endfield-gacha-phase1-design.md`. Rules and assumptions:
`docs/assumptions.md`. Research direction: `docs/research-notes.md`. Numbers: `docs/results.md`.

## Commands

    uv sync                      # environment
    uv run pytest                # all tests (paper reproduction + Monte Carlo parity included)
    uv run ruff check . && uv run ruff format .
    uv run gacha evaluate --pity 40 --banner-pulls 20 --budget 60 --realized 90
    uv run gacha experiment all  # tables -> results/tables, figures -> results/figures

## Principles

- **One rule source, two engines.** Rules live in `src/gacha/rules/` (parameters) and
  `src/gacha/models/` (transitions). The exact engine (`kernel/`) and Monte Carlo (`mc/`) both
  consume `Model.step()`; never re-implement a rule in a simulator or an experiment.
- **T = paid pulls.** Free pulls are cost-0 transitions. Budgets, quantiles and VaR are in paid pulls.
- **Every model is a finite DAG.** Each transition must strictly increase a counter; the chain
  enumerator raises `CycleError` otherwise.
- **Assumptions are registered.** Any rule or policy change updates `docs/assumptions.md` first,
  then code, tests and affected experiments (use the `add-mechanic` skill).
- **Paper reproduction stays green.** `tests/test_paper_reproduction.py` pins the paper's numbers.
  Golden values (`tests/test_endfield_model.py::GOLDEN`) change only with a justification in the
  commit message and a note in `docs/research-notes.md`.
- **Experiments are reproducible scripts.** `src/gacha/experiments/eXX_name.py` exposes
  `run(out_dir) -> dict`; register it in `EXPERIMENTS`; write tables (CSV + MD) and figures (PNG +
  PDF); return headline numbers (use the `run-experiment` skill).
- **Figures** use `gacha.plots.style.PALETTE` in slot order, one axis per chart, a legend whenever
  two or more series are drawn, axis labels with units ("paid pulls").

## Layout

    src/gacha/rules      parameters (schedule.py, paper.py, endfield.py)
    src/gacha/kernel     types, chain enumeration, forward (PMF), backward (E, Var, success within b)
    src/gacha/models     paper.py, endfield.py (single banner), plan.py (multi-banner)
    src/gacha/risk       metrics.py, normal.py
    src/gacha/mc         simulate.py, compare.py
    src/gacha/plots      style.py, waiting_time.py, heatmaps.py
    src/gacha/experiments  registry + e01..e07
    tests/               one file per module; toy_models.py holds hand-checkable fixtures
```

- [ ] **Step 4: Write the two skills (invoke `superpowers:writing-skills` first)**

`.claude/skills/add-mechanic/SKILL.md`:
```markdown
---
name: add-mechanic
description: Use when adding or changing a gacha rule (a new milestone, a rate change, a new currency effect, a new banner type) in this repo - enforces the register-first, test-first, exact-vs-Monte-Carlo workflow.
---

# Add or change a game mechanic

## Checklist

1. **Register the rule.** Add or edit the row in `docs/assumptions.md` with source and status.
2. **Add the parameter** to the rules dataclass (`src/gacha/rules/`) with a default that keeps
   current behavior; validate it in `__post_init__`.
3. **Write failing tests first** (`tests/`):
   - a rule unit test on the parameter or schedule value;
   - a transition test on `pull()`/`step()` asserting exact branch probabilities;
   - a closed-form or hand-computed case where possible (small caps, `up_share=1`, `hard_pity=1`);
   - a Monte Carlo parity case in `tests/test_mc.py` if the state space changed.
4. **Implement** in `src/gacha/models/` only. Keep every transition strictly increasing a counter.
5. **Run** `uv run pytest`. Paper reproduction must stay green. If a golden value moves, justify
   it in the commit message and in `docs/research-notes.md` (Findings).
6. **Rerun affected experiments** (`uv run gacha experiment <name>`) and refresh `docs/results.md`.
7. **Record research impact** in `docs/research-notes.md` if a proposition is affected.

## Do not

- Re-implement a rule inside `mc/`, `experiments/` or `cli.py`.
- Silence a failing reproduction or monotonicity test by editing its expected value.
```

`.claude/skills/run-experiment/SKILL.md`:
```markdown
---
name: run-experiment
description: Use when running, adding, or modifying an experiment script in this repo (tables, figures, headline numbers in docs/results.md) - covers registry, output conventions and figure style.
---

# Run or add an experiment

## Run

    uv run gacha experiment <name> [--out results] [--results-md docs/results.md]
    uv run gacha experiment all

Tables land in `results/tables/<name>_*.csv|.md`, figures in `results/figures/<name>_fig_*.png|.pdf`,
headline numbers replace the `## <name>` section of `docs/results.md`. `results/` is git-ignored;
`docs/results.md` is committed.

## Add

1. Create `src/gacha/experiments/eNN_short_name.py` with `run(out_dir: Path) -> dict[str, float]`.
2. Build models from `gacha.rules` / `gacha.models`; compute with `gacha.kernel`; summarize with
   `gacha.risk.metrics`; never hard-code rule values.
3. Write every table with `write_table(df, out_dir, "eNN_...")` and every figure with
   `save(fig, figures_dir, "eNN_fig_...")` after `apply_style()`.
4. Append the name to `EXPERIMENTS` in `src/gacha/experiments/__init__.py` (keep order).
5. Add a smoke test in `tests/test_experiments.py`: run into `tmp_path`, assert files exist and
   two or three headline numbers satisfy a sanity inequality.
6. Run it, open the PNGs, check labels and legends, then commit code + test + `docs/results.md`.

## Figure rules

- Palette from `gacha.plots.style.PALETTE`, slot order fixed; at most 4 series per axis.
- One y axis per chart; heatmaps use the single-hue `Blues` map.
- Legend whenever two or more series are drawn; axis labels carry units ("paid pulls").
- Title states what is plotted, not a conclusion.
```

- [ ] **Step 5: Run everything and refresh results**

Run:
```bash
uv run ruff check . && uv run ruff format --check . || uv run ruff format .
uv run pytest
uv run gacha experiment all --out results --results-md docs/results.md
```
Expected: lint clean, all tests pass, `docs/results.md` has seven `##` sections. Read `docs/results.md` and copy any surprising number (for example a monotonicity flag of 0, or E4 gaps) into the Findings section of `docs/research-notes.md`.

- [ ] **Step 6: Commit**

```bash
git add docs/assumptions.md docs/research-notes.md docs/results.md CLAUDE.md .claude/skills
git commit -m "docs: assumptions register, research notes, CLAUDE.md, project skills and Phase 1 results

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Definition of done (spec §10)

- `uv run pytest` green, including paper reproduction and Monte Carlo parity.
- `uv run gacha experiment all` produces every table and figure of Tasks 12-14 without error.
- `docs/assumptions.md`, `docs/research-notes.md`, `docs/results.md`, `CLAUDE.md` and the two
  skills exist and match the code.
