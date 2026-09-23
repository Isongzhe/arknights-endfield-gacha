---
name: add-mechanic
description: Use when adding or changing any gacha game rule or player policy in this repo - a milestone (30/60/120/240), a rate, a default of EndfieldCharacterRules, a new banner type or currency effect - before touching src/gacha/rules or src/gacha/models.
---

# Add or change a game mechanic

## Overview

One rule source, two engines: the exact kernel and Monte Carlo both read `Model.step()`.
A rule change is a registered, test-first change that ends with refreshed experiments.
The order below is the contract; skipping a step is the failure, however small the change.

## Checklist (in this order)

1. **Register.** Edit the row in `docs/assumptions.md` (rule, source, status, parameter).
   A new rule gets a new ID; an open question gets updated.
2. **Parameter.** Add or change it in `src/gacha/rules/*.py`. A new parameter defaults to the
   current behavior and is validated in `__post_init__`.
3. **Failing tests first.** Write them, run them, read the failure before any model code:
   - `tests/test_rules.py`: the new parameter or schedule value;
   - `tests/test_endfield_model.py`: a `pull()` / `status()` branch test with exact
     probabilities at the new trigger state;
   - a closed form or hand-computed case when one exists (small cap, `up_share=1`,
     `hard_pity=1`, `n` just before the trigger).
4. **Implement** in `src/gacha/models/` only. Every transition must strictly increase a counter.
5. **Run `uv run pytest`.** Paper reproduction (`tests/test_paper_reproduction.py`) stays green.
6. **Golden values moved?** (`tests/test_endfield_model.py::GOLDEN`) Recompute them from the
   model, paste them, and write the reason in BOTH the commit message and the Findings section
   of `docs/research-notes.md`.
7. **Rerun affected experiments** and commit the refreshed `docs/results.md`:
   `uv run gacha experiment all` (E2, E3, E5, E6, E7 use the default rules).
8. **Research impact.** If a proposition in `docs/research-notes.md` mentions the mechanic,
   update its text.

## Rationalizations seen in this repo

| Excuse | Reality |
|--------|---------|
| "Existing tests already cover the mechanism; I'll fix expectations after changing the code" | Expectations edited after the code only record what the code does. Write the new expected value first and watch it fail. |
| "Rerunning experiments isn't needed for pytest to pass" | `docs/results.md` is committed and is now stale. Rerun and commit it. |
| "It's just a default-parameter change" | Defaults feed GOLDEN, E2/E3/E5/E6/E7 and `docs/assumptions.md`. Same checklist. |
| "results/ is git-ignored, so nothing goes stale" | The figures are ignored; the headline numbers in `docs/results.md` are not. |

## Red flags - stop and go back to step 1

- Editing `src/gacha/rules` or `src/gacha/models` before `docs/assumptions.md` has the row
- Changing a number in a test so it passes, without a failing run first
- A GOLDEN change whose reason is not in the commit message
- A rule constant (0.008, 120, 30, ...) typed into `mc/`, `experiments/` or `cli.py`
