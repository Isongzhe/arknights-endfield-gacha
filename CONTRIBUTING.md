# Contributing

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format .
```

## Changing or adding a game rule

Rules decide every number, so they follow a fixed order:

1. Record the rule, its source and its status in `docs/assumptions.md`.
2. Add or change the parameter in `src/gacharisk/rules/`. A new parameter defaults to current behaviour.
3. Write a failing test first: the parameter, the exact branch probabilities of the transition,
   and a hand-checkable case where one exists.
4. Implement in `src/gacharisk/models/` only. The simulator and the experiments must not restate rules.
5. Run the suite. The paper reproduction tests must stay green.
6. If the frozen values in `tests/test_endfield_model.py` move, recompute them and explain why in
   the commit message and in `docs/research-notes.md`.
7. Rerun the experiments (`uv run gacha-risk experiment all`) and commit `docs/results.md`.

## Adding another game

Add a rules dataclass and a model exposing `initial()` and `step(state)`; every transition must
strictly advance a counter so the state graph stays acyclic. The kernel, risk metrics and
simulator then work unchanged. Add it to `tests/test_kernel_invariants.py`.

## Sources

Cite where a rule comes from. Official in-game text beats wikis, and wikis beat guides. If the
game publishes aggregate rates, add a calibration check like the one in `docs/assumptions.md`.
