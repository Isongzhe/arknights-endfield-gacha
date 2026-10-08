# Architecture

```
rules/      parameters only: rates, pity, guarantees, bonuses (dataclasses)
models/     state and one-step transitions built from rules
kernel/     enumerate a model, forward distribution, backward expectations
risk/       metrics on a distribution: quantiles, VaR/CVaR, completion, shortfall
analysis/   decisions built on distributions: two banners one stock, rebate, arsenal income
cost/       price menus: cheapest purchase for a number of pulls (data, not rules)
mc/         Monte Carlo on the enumerated transitions, for checking only
experiments/  reproducible scripts that write tables and figures
cli.py, scenario.py   command line and TOML inputs
```

## The one contract

A model exposes `initial()` and `step(state) -> list[Transition]`. A transition has a probability,
a next state or an absorbing label (success or fail), and a cost of 0 or 1. Every transition must
advance a counter, so the reachable states form a finite directed acyclic graph.

Everything else follows from that contract:

- `kernel.chain` enumerates the states once and stores the transitions split by cost.
- `kernel.forward` propagates probability mass level by level, where a level is the number of
  cost-1 moves made so far. Cost-0 moves (free pulls) are closed within a level. The result is
  the exact distribution of the cost until absorption.
- `kernel.backward` walks the graph in reverse topological order for expectations, variances and
  the probability of success within a budget, for every state at once.
- `mc.simulate` samples the same stored transitions, so the simulator cannot disagree with the
  exact engine about the rules.

What "cost" means is the model's choice: one own pull for character banners, one ten-pull issue
for the weapon banner.

## What kind of model this is

Every model here is a Markov chain evaluated under a fixed policy (for example "pull until the
target or until the cap"). The engine computes the exact distribution of the outcome of that
policy; comparing policies means evaluating each one. It is not yet a Markov decision process:
nothing in the code optimises over actions. Posing the budgeted problem as an MDP on the same
state space is the planned next step (`docs/theory-problems.md`, P5).

## Rules, data and state are kept apart

| Kind | Changes | Lives in |
|---|---|---|
| Game rules | rarely | `src/gacharisk/rules/`, registered with source and status in `docs/assumptions.md` |
| Game data (schedule, roster, prices) | every version | `docs/site/*.json`, `docs/rules/`, `gacharisk.cost` |
| Player state (resources, counters, progress) | every pull | inputs: CLI flags, scenario TOML, the site's saved form |

A rule change goes register first, failing test second, model third (see CONTRIBUTING.md).

## Two engines on the site

`docs/site/engine.js` ports three recursions (first rate-up copy, consecutive limited banners,
weapon banner) so the calculators run in the browser. `docs/site/build.py` compares each port with
the Python engine on fixed cases and refuses to build if they differ. Python is the reference.

## Verification layers

1. Reproduction of the reference paper's published numbers: the engine.
2. Calibration against the operator's published comprehensive rates: the rules.
3. Monte Carlo parity within three standard errors: the implementation.
4. Hand-computable cases and kernel invariants on every model: regressions.
