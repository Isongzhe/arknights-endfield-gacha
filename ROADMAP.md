# Roadmap

The project has two goals that share one codebase: a **tool** players can use to plan pulls from
their current state, and a **paper** on exact waiting-time and budget-risk analysis of multi-counter
pity systems. Work is ordered so that each step serves both.

## Done (0.2.0)

- Exact engine: state enumeration, forward distribution over own pulls, backward expectations.
- Models: reference-paper models, Endfield limited and re-run character banners, multi-banner
  plans with carried pity, weapon banner (first rate-up copy).
- Validation: reference-paper reproduction, calibration against the official comprehensive rates,
  Monte Carlo parity, kernel invariants on every model.
- Analysis: stopping table for two banners sharing one stock, money risk through a price menu,
  保障配額 rebate (expected-value layer), arsenal income.
- Tool surface: CLI (`evaluate`, `decide`, `weapon`, `experiment`), documentation
  site with calculators.
- Data: rule register, operator roster, past banner schedule, price list.

## Next: make the tool state-driven

| Step | Why it matters for the tool | Why it matters for the paper |
|---|---|---|
| One player-state file (resources, pity counters, banner progress, ownership, offers left, spending cap) read by CLI, notebook and site | Conditions change after every pull; update the state instead of re-entering inputs | Defines the state space the decision problem is posed on |
| "Record an outcome" (spent N pulls, got X) that updates the state | Daily use without an assistant | Yields pull logs for empirical checks |
| One plan format: a list of banners with type, dates, target and cap | Replaces the two fixed calculator modes | The object the optimal policy is computed for |
| Banner schedule and roster as the single data source | Off-rate pool and end dates derived, not typed | Reproducible inputs |
| Long-horizon view as rates (income per version vs pulls per character), not joint probabilities | Honest about what cannot be known weeks ahead | Separates exact results from planning heuristics |

## Then: the decision problem (paper core)

1. Budget-constrained optimal stopping as a finite MDP over the same state space; adaptive
   policies, including whether to use free pulls on a skipped banner.
2. Proofs: stopping points lie on pity thresholds; monotone value of pity; stochastic ordering in
   the guarantee position.
3. Hybrid rebate: the 6★ part of the 保障配額 rebate inside the exact chain.
4. A second game with a different mechanism, or an abstract definition of the model class, to
   support the claim of generality.
5. Empirical check against recorded pull histories, if data becomes available.

## Not planned

- Player-behaviour, harm or revenue modelling: no data, and outside the scope of an exact model.
- A hosted backend: every computation runs in milliseconds in the browser.

See [docs/theory-problems.md](docs/theory-problems.md) for the problems stated formally,
[docs/paper-outline.md](docs/paper-outline.md) for the paper plan and
[docs/research-notes.md](docs/research-notes.md) for dated findings.
