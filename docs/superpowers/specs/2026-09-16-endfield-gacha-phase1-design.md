# Endfield Gacha Waiting-Time Models — Phase 1 Design

Date: 2026-09-16
Status: approved design, awaiting spec review
Scope: Phase 1 of a research codebase (exact character-banner model + risk metrics + Monte Carlo validation + experiments). Phases 2–3 are sketched only so the architecture accommodates them.

## 1. Background and motivation

Reference paper: Hou, S.; Zhu, Y.; Zhang, S. *State-Dependent Asymmetry in Soft-Pity Gacha Waiting-Time Models: Exact Recurrences, Tail Risk, and Featured-Target Extensions.* Symmetry 2026, 18(6), 1051. DOI 10.3390/sym18061051. CC BY 4.0. Local copy: obtain from `https://mdpi-res.com/d_attachment/symmetry/symmetry-18-01051/article_deploy/symmetry-18-01051.pdf`.

What the paper does:

- Models the waiting time for one rare item as the absorption time of a finite absorbing Markov chain whose only transient state is the pity counter t. Q is skip-free upper-bidiagonal, hence nilpotent.
- Derives first-step recurrences: E_t = 1 + (1−p_t)E_{t+1}; V_t = (1−p_t)V_{t+1} + p_t(1−p_t)E²_{t+1}; PMF f_t(1) = p_t, f_t(j) = (1−p_t)f_{t+1}(j−1).
- Proves a stochastic-ordering proposition (statewise hazard increase ⇒ usual stochastic order).
- Obtains multi-item distributions by i.i.d. convolution (its Eq. 2), then reports quantiles, VaR/CVaR, expected excess, completion probability, skewness, entropy, normal-approximation error, sensitivity analyses. Monte Carlo is used only as a numerical check.
- Adds a featured-target variant with a binary guarantee state (t, g): a 50/50 whose loss makes the next rare item guaranteed featured (Genshin-style).
- Explicitly lists as future work: featured guarantees, duplicate collection, cross-banner carry-over, budget-dependent stopping, currency conversion, banner-specific rules.

Our direction: apply the same "exact recurrences, no simulation error" philosophy to Arknights: Endfield (明日方舟：終末地), whose limited character banner has a structurally different asymmetry from the paper's (t, g) model: **two counters plus a one-time, expiring guarantee**. Later phases add monetary cost and budget-constrained decisions (stop-loss, multi-banner scheduling) and the weapon banner.

## 2. Phase 1 scope

In scope:

1. Exact waiting-time distributions (in paid pulls) for the Endfield limited character banner under the full rule set in §3, from any initial state.
2. Objectives: first UP copy, or k UP copies (potentials), within one banner.
3. Evaluation of a *given* multi-banner plan (sequence of banners with per-banner paid-pull caps and target copies), including cross-banner carry-over of pity and the 60-pull dossier.
4. Risk metrics (§5.8) on possibly defective distributions.
5. Monte Carlo validation that consumes the same rule definition as the exact engine.
6. Reproduction of the paper's numbers as regression tests (single counter, hard-pity-only, featured 50/50).
7. Experiment scripts producing tables and figures (§7).
8. A small CLI: evaluate a personal state, run experiments.
9. Project skills for Claude Code and a CLAUDE.md.

Out of scope (later phases; architecture leaves hooks):

- Phase 2: TWD cost curve and purchase menu (double-bonus stones, packs, monthly cards), monetary VaR/CVaR, resource income model, optimization of per-banner caps (chance-constrained DP / stop-loss), personal luck history import.
- Phase 3: weapon banner (Arsenal Exchange), 保障配額 / 集成配額 currencies and their conversion into pulls, 5★ layer, off-rate 6★ composition (standard pool vs previous two limited operators), collection objectives over the mixed pool.

## 3. Domain rules and assumptions register

Sources: user statement (2026-09-16), English wiki `endfield.wiki.gg/wiki/Headhunting`, Chinese guides (gamersky, 17173). Status "confirmed" means user and wiki agree; "assumed" means a modeling choice to verify with the user. Every rule maps to a parameter of `EndfieldCharacterRules` (§5.5) so it can be toggled. The register is copied to `docs/assumptions.md` at implementation time and kept current.

| ID | Rule | Status | Parameter |
|----|------|--------|-----------|
| R1 | One pull costs 500 合成玉; 1 源石 = 75 合成玉. Not used in Phase 1 (cost layer). | confirmed | Phase 2 |
| R2 | 6★ probability depends on t, the number of consecutive non-6★ counted pulls since the last 6★: p_t = 0.008 for 0 ≤ t ≤ 64; p_t = 0.008 + 0.05·(t − 64) for 65 ≤ t ≤ 78; p_79 = 1 (hard pity on the 80th pull). Checks: p_65 = 0.058, p_78 = 0.708. | confirmed | base_rate, soft_pity_start=65, soft_pity_step, hard_pity=80 |
| R3 | Any 6★ from a counted pull resets t to 0. t carries over between limited character banners. | confirmed | — |
| R4 | Each 6★ from a counted pull is the current UP with probability q = 0.5, independently of history. There is no "next 6★ guaranteed UP after a loss" state. | confirmed | up_share |
| R5 | Banner-local guarantee: if the UP has not yet been obtained from counted pulls in this banner (u = 0) and the counted pull is the 120th of the banner (n reaches 120), that pull is the UP with probability 1. One-time per banner; void once u = 1; not carried over. The guaranteed pull is a 6★ and resets t. | confirmed (pity reset: assumed) | guarantee_pull=120 |
| R6 | 30-pull bonus: when n reaches 30, the player receives 10 free "vacuum" pulls on the current banner. They neither read nor modify t, n, u, and a UP obtained from them does not void the 120 guarantee. Each vacuum pull is a 6★ with probability 0.008 (flat) and UP with probability q given 6★. Copies obtained add to c. | isolation confirmed; flat rate assumed | vacuum_at=30, vacuum_pulls=10, vacuum_rate=0.008 |
| R7 | 60-pull dossier: if n ≥ 60 at the end of banner k, the player receives 10 free counted pulls at the start of the chronologically next limited banner k+1. They are ordinary pulls (advance t and n, benefit from soft pity, can trigger everything). If banner k+1 is skipped they are still used there (they cost nothing and advance t). | confirmed; counting in n assumed | dossier_at=60, dossier_pulls=10 |
| R8 | 240-pull potential: each time n reaches a multiple of 240, one extra UP copy is granted (c += 1). Not carried over. | confirmed | potential_every=240 |
| R9 | Each banner grants 5 free banner-bound pulls at its start (given in the first three days of the version). Modeled as the first 5 counted pulls, cost 0. If these turn out to be generic permits rather than banner-bound, they belong in the Phase 2 income model instead; parameter default is then 0. | user; banner-bound assumed | free_start_pulls=5 |
| R10 | Off-rate 6★ composition (standard pool vs the previous two limited operators mixed in). Ignored in Phase 1; the transition only labels the outcome OFF. | deferred | Phase 3 |
| R11 | 5★ layer: 8% rate, at least one 5★-or-above per 10 pulls. Does not influence 6★ outcomes. Ignored in Phase 1. | deferred | Phase 3 |
| R12 | A limited operator appears in three consecutive banners (own banner, then mixed into the next two). Only affects R10. | deferred | Phase 3 |

Policy assumptions (about the player, not the game):

| ID | Assumption |
|----|------------|
| P1 | Free pulls are used before paid pulls within a banner (rational; also the dossier is consumed by the first headhunt on that banner). Hence paid pulls in a banner = max(0, n − f) where f = free_start_pulls + dossier_pulls·d. |
| P2 | Free pulls are always used, even on a skipped banner, because they cost nothing and advance t. |
| P3 | On a wanted banner the player pulls until the target copies are obtained or the paid cap x is exhausted. If the target is reached while free pulls remain, the remaining free pulls are still used, then paid pulling stops. |
| P4 | Waiting time T counts paid pulls only. Free pulls affect the state but not T. |

## 4. Approach

Chosen approach (B): a generic finite absorbing-chain engine plus composable rule models. A model defines a hashable state and a one-step transition function; the engine enumerates reachable states, computes the hitting-time distribution over paid pulls forward, computes expectations and variances backward on the DAG, and composes banners into plans. Monte Carlo samples the same transition function, so the rules exist in exactly one place. The paper's models are instances of the same protocol and serve as regression tests.

Rejected: (A) hand-written recurrences per model (paper style) — transparent but every new mechanic requires re-deriving recurrences and duplicating rules in the simulator; (C) generating functions — useful for the paper's mathematics on the single-counter case, unsuitable as the code's backbone because the banner-local one-time guarantee makes closed forms unwieldy.

## 5. Detailed design

### 5.1 Kernel abstractions (`gacha.kernel`)

```python
State = Hashable              # small tuples in practice

@dataclass(frozen=True)
class Transition:
    prob: float               # transitions from one state sum to 1
    next: State | None        # None when absorbed
    cost: int                 # paid pulls consumed by this transition: 0 or 1
    absorb: str | None        # "success" | "fail" | None

class Model(Protocol):
    def initial(self) -> State: ...
    def step(self, s: State) -> list[Transition]: ...
```

Requirements on models: the state graph reachable from the initial states must be finite and acyclic (every transition strictly increases a counter such as banner-local pulls or banner index). The kernel enumerates states by BFS, merges duplicate `(next, cost, absorb)` targets by summing probabilities, and detects cycles with Kahn's algorithm, raising a clear error if one exists. Anything an experiment wants to count (UP copies, 6★ events) must be part of the state; Phase 1 needs only c.

`EnumeratedChain.from_model(model, roots=None)` enumerates from `model.initial()` or from an explicit list of root states (needed to evaluate an initial *distribution* over entering states). It holds: state list and index map; CSR-style transition arrays split by cost (`Q0`, `Q1` as `scipy.sparse.csr_matrix`), absorption vectors `a0_succ, a0_fail, a1_succ, a1_fail`, the topological order, and `max_cost_path`, the longest cost-weighted path in the DAG (computed in topological order), which is the exact support bound of T and the default forward horizon.

### 5.2 Forward algorithm: hitting-time distribution over paid pulls

Input: initial distribution μ₀ over states (a point mass by default; any distribution to evaluate uncertain entering states), horizon J (max paid pulls, default `max_cost_path`, so the residual is zero unless a smaller J is requested).

For each budget level j = 0..J:

1. Closure of free moves: W = Σ_{i≥0} μ_j Q0^i, computed by iterating `v ← v·Q0` until v is exactly zero (Q0 is nilpotent because the graph is a DAG). W is the visit measure at level j.
2. Absorption at level j through free transitions: `f_succ[j] += W·a0_succ`, `f_fail[j] += W·a0_fail`.
3. Absorption at level j+1 through the next paid pull: `f_succ[j+1] += W·a1_succ`, `f_fail[j+1] += W·a1_fail`.
4. Advance: μ_{j+1} = W·Q1.

Mass conservation holds exactly: μ_{j+1}·1 + (absorbed at j+1 via paid) = μ_j·1 − (absorbed at j via free). Outputs: `f_succ`, `f_fail` (arrays over j), and the residual unabsorbed measure μ_{J+1} (used to report, e.g., the distribution of pity t after exhausting a budget). A `HittingTime` result type wraps these with helpers (`p_success`, `pmf_success`, `cdf`, `survival`).

### 5.3 Backward algorithm: expectations per state

In reverse topological order, for each state s with transitions {(p, next, cost, absorb)}:

- `E[s] = Σ p·(cost + E[next])` with E[None] = 0.
- `M2[s] = Σ p·(cost² + 2·cost·E[next] + M2[next])`; `Var[s] = M2[s] − E[s]²`.
- `Psucc[s] = Σ p·(1{absorb == "success"} + Psucc[next])`.

For the paper's single-counter model this reduces exactly to its Propositions 1–2; a test checks the second-moment form equals the law-of-total-variance form. E and Var here are for T = paid pulls until absorption of either kind.

### 5.4 Paper models (`gacha.models.paper`)

- `PaperSchedule`: p_t = base for t < soft_start; base + (t − soft_start + 1)·slope for soft_start ≤ t < hard − 1; p_{hard−1} = 1. Representative schedule: base 0.006, soft_start 73, slope 0.0585, hard 90 (paper Eq. 1). Hard-pity-only variant: slope 0.
- `SingleCounterModel(schedule)`: state (t,); success with p_t, else (t+1,); every transition cost 1.
- `Featured5050Model(schedule, q=0.5)`: state (t, g); paper Eq. (21)–(22); repeated targets via `IIDStagesModel(model, m)` whose state is (stage, inner) and which restarts the inner model from its initial state after each success, matching the paper's reset convention.
- `risk.convolve(pmf, m)` provides the paper's i.i.d. convolution independently of the model layer.

### 5.5 Endfield character banner (`gacha.rules.endfield`, `gacha.models.endfield`)

```python
@dataclass(frozen=True)
class EndfieldCharacterRules:
    base_rate: float = 0.008
    soft_pity_start: int = 65      # first t at which the ramp applies (the 66th pull)
    soft_pity_step: float = 0.05
    hard_pity: int = 80            # p_{hard_pity-1} = 1
    up_share: float = 0.5
    guarantee_pull: int | None = 120
    vacuum_at: int | None = 30
    vacuum_pulls: int = 10
    vacuum_rate: float = 0.008
    dossier_at: int | None = 60
    dossier_pulls: int = 10
    potential_every: int | None = 240
    free_start_pulls: int = 5

    def p(self, t: int) -> float: ...   # R2
```

Banner instance: `BannerSpec(rules, target_copies: int, cap: int)` where `target_copies = 0` means a skipped banner (only free pulls are used; `cap` must be 0) and `cap` is the maximum number of paid pulls.

Banner state (t, n, c, u):

- t ∈ [0, hard_pity): pity counter, carried in from the previous banner.
- n ≥ 0: counted pulls in this banner (free and paid). Paid so far = max(0, n − f).
- c: UP copies obtained in this banner, capped at target_copies (states beyond the cap are merged).
- u ∈ {0, 1}: whether a UP has been obtained from a counted pull (voids the 120 guarantee).

`pull(state, f)` returns the next-state distribution for one counted pull, with `cost = 0 if n < f else 1` and n' = n + 1:

1. If `guarantee_pull` is set, u = 0 and n' = guarantee_pull: forced UP with probability 1 → t' = 0, c' = c + 1, u' = 1.
2. Otherwise with p = p_t: 6★. Given 6★: UP with probability q → (0, n', c + 1, 1); OFF with probability 1 − q → (0, n', c, u). With 1 − p: (t + 1, n', c, u).
3. Post-processing on every branch: if `potential_every` divides n' → c' += 1. If n' = vacuum_at → split the branch by k ~ Binomial(vacuum_pulls, vacuum_rate·q) copies, c' += k.
4. Cap c' at target_copies and merge identical next states.

`status(state, spec, f)`: `"success"` if c ≥ target_copies ≥ 1 and n ≥ f (P3); `"exhausted"` if paid ≥ cap and not success (for skipped banners this is simply n ≥ f); else `"active"`.

`SingleBannerModel(spec, start=BannerState(t=0, n=0, c=0, u=0), dossier=False)`: wraps a banner; absorbs `success` on status success and `fail` on exhausted. Absorbing transitions from terminal statuses have cost 0 and probability 1. `start` allows evaluation from a mid-banner personal state (§5.9).

### 5.6 Multi-banner plan (`gacha.models.plan`)

`Plan(banners: list[BannerSpec], start=BannerState(0, 0, 0, 0), dossier0: bool = False)`. State (k, t, n, c, u, d) where d is the dossier flag entering banner k (f_k = free_start_pulls + dossier_pulls·d). `start` is the state within the first banner (fresh by default; a personal mid-banner state otherwise).

- Active banner: transitions from `pull` (§5.5).
- Banner k ends with status success, or exhausted on a skipped banner: if k is the last banner → absorb `success`; else boundary transition (cost 0, prob 1) to (k+1, t, 0, 0, 0, d' = 1{n ≥ dossier_at}).
- Exhausted on a wanted banner → absorb `fail` (the plan's all-targets objective is already lost; P(success) is what Phase 2's chance-constrained objective maximizes).

The paper's i.i.d. convolution is recovered when every banner starts from the same state, which is false whenever t or d carries over; experiment E4 quantifies the difference.

### 5.7 Monte Carlo (`gacha.mc`)

`simulate(chain: EnumeratedChain, n_paths: int, seed: int) -> Paths` runs a vectorized simulation on the enumerated index representation (per-state cumulative transition probabilities, `searchsorted` sampling), returning per path: absorb label, paid pulls, UP copies gained. `compare(exact: HittingTime, paths: Paths, thresholds)` reports P(success), mean, and tail probabilities with Monte Carlo standard errors and a pass/fail against a tolerance in standard errors (default 3).

### 5.8 Risk metrics (`gacha.risk`)

All take a `HittingTime`. Two distributions live in it and every metric names which one it uses through a `which` argument:

- `"stop"` (default): T_stop = paid pulls until absorption of either kind, `pmf_stop = f_succ + f_fail`. A proper distribution (mass 1 when the horizon covers the support). Spend is spend, so budget and tail-risk questions use this one.
- `"success"`: the success-only PMF `f_succ` with mass P(success) ≤ 1; metrics on it are conditional on success (normalized by P(success)). When P(success) = 1 the two coincide, which is the paper's situation.

Functions:

- `p_success`, `mean`, `var`, `sd`, `cv`.
- `cdf(b)`, `survival(b)`; `completion(b) = P(success ∧ T ≤ b)` always uses `f_succ` unnormalized.
- `quantile(alpha)`: smallest j with CDF ≥ α. `var_at(alpha)` is an alias. `cvar(alpha) = E[T | T ≥ VaR_α]` (discrete, as in the paper).
- `expected_excess(b) = E[(T − b)⁺]`.
- `skewness`, `entropy_nats`.
- `normal_approx_cdf(b)` with continuity correction and `max_abs_cdf_error`.
- `convolve(pmf, m)` for i.i.d. stages.
- `summary(ht, alphas=(0.5, 0.75, 0.9, 0.95, 0.99), budgets=...)` returning a tidy `pandas.DataFrame` for tables.

### 5.9 CLI (`gacha.cli`)

- `gacha evaluate --pity T --banner-pulls N --copies C --up-obtained 0|1 --dossier 0|1 --target M --cap X --budget B [--realized J]` builds `SingleBannerModel` with `start=(T, N, C, U)` and prints P(success within B), E, sd, quantiles, VaR/CVaR, expected excess at B, and, with `--realized J`, the survival probability at J from a fresh state (how unlucky a realized outcome was). `--plan plan.toml` evaluates a multi-banner plan instead (TOML via stdlib `tomllib`; schema: `t0`, `dossier0`, `[[banners]]` with `target_copies`, `cap`, optional rule overrides).
- `gacha experiment <name> [--out results/]` runs one registered experiment; `gacha experiment all` runs every experiment.

### 5.10 Project layout and tooling

```
gacha/
├── pyproject.toml            # uv-managed; deps: numpy, scipy, matplotlib, pandas; dev: pytest, ruff
├── uv.lock
├── .gitignore                # results/, .venv/, __pycache__, *.pdf figures are regenerated
├── CLAUDE.md
├── .claude/skills/
│   ├── add-mechanic/SKILL.md
│   └── run-experiment/SKILL.md
├── src/gacha/
│   ├── __init__.py
│   ├── rules/__init__.py, endfield.py, paper.py
│   ├── kernel/__init__.py, chain.py (enumeration, DAG check), forward.py, backward.py, types.py
│   ├── models/__init__.py, paper.py, endfield.py, plan.py
│   ├── risk/__init__.py, metrics.py, normal.py
│   ├── mc/__init__.py, simulate.py, compare.py
│   ├── plots/__init__.py, style.py, waiting_time.py, heatmaps.py
│   └── cli.py
├── experiments/
│   ├── __init__.py (registry)
│   ├── e01_reproduce_paper.py … e07_sensitivity.py
├── results/                  # git-ignored: tables/*.csv|*.md, figures/*.png|*.pdf
├── tests/
│   ├── test_rules_endfield.py, test_rules_paper.py
│   ├── test_kernel_properties.py
│   ├── test_paper_reproduction.py
│   ├── test_endfield_model.py, test_plan.py
│   ├── test_risk.py
│   └── test_mc_parity.py
└── docs/
    ├── superpowers/specs/2026-09-16-endfield-gacha-phase1-design.md
    ├── assumptions.md        # the register in §3, kept current
    ├── research-notes.md     # contributions vs Hou et al., propositions to derive (§8)
    └── results.md            # headline numbers, updated by experiments
```

Toolchain: Python ≥ 3.12, `uv` for environment and lock file, `ruff` for lint/format, `pytest`. Commands: `uv sync`, `uv run pytest`, `uv run ruff check .`, `uv run gacha ...`.

### 5.11 CLAUDE.md content

Project purpose and paper reference; the one-rule-source principle (exact engine and Monte Carlo share `step()`); where rules and assumptions live and that every rule change must update `docs/assumptions.md`; commands; the requirement that paper-reproduction tests stay green and that golden values change only with a justification recorded in the commit; experiment conventions (numbering, outputs, figure style); pointers to the two skills.

### 5.12 Skills

Written with the `superpowers:writing-skills` skill during implementation. Keep each under ~80 lines.

- `add-mechanic`: checklist for adding or changing a game rule: record the rule and its source in `docs/assumptions.md`; add a parameter with a default that preserves current behavior; write failing tests first (rule unit test, kernel property test, Monte Carlo parity); implement in the model's `pull`/`step`; rerun affected experiments; update golden values only with justification; update `docs/research-notes.md` if a proposition is affected.
- `run-experiment`: how to run and add experiments (registry entry, `run(out_dir)` signature, table and figure naming), figure style rules (consistent palette, labeled axes with units "paid pulls", PNG and PDF), and updating `docs/results.md` with headline numbers.

## 6. Testing strategy

Rules:

- Endfield p_t: p_64 = 0.008, p_65 = 0.058, p_78 = 0.708, p_79 = 1; length 80.
- Paper schedule: p_72 = 0.006, p_73 = 0.0645, p_88 = 0.006 + 16·0.0585 = 0.942, p_89 = 1.

Kernel properties (on every model fixture):

- Transition probabilities from each state sum to 1 within 1e−12.
- `f_succ.sum() + f_fail.sum() + residual == 1` within 1e−12 when J is large enough that the residual is 0.
- Backward E and Var equal the moments computed from the forward PMF within 1e−9.
- Second-moment variance equals the paper's law-of-total-variance recurrence on the single-counter model.
- Cycle detection raises on a deliberately cyclic toy model.

Paper reproduction (`test_paper_reproduction.py`, tolerance 0.01 on moments, exact on integer quantiles, 5e−4 on probabilities):

- Single counter: E₀ = 62.34, V₀ = 592.43, sd = 24.34; E₇₀ = 7.68, E₇₂ = 5.76, E₇₃ = 4.78, E₈₀ = 1.92; hard-pity-only E₀ = 69.70.
- T₁₀ by convolution: mean 623.38, var 5924.34; quantiles 50/75/90/95/99% = 629/679/719/741/775; P(T₁₀ ≥ 724) = 0.0876; P(T₁₀ ≥ 778) = 0.0073; CVaR₉₀ = 744.66; expected excess at 700 = 5.11; completion at 719 = 0.9014.
- Distributional asymmetry: soft-pity skew −1.24, entropy 3.61 nats; hard-only skew −1.08, entropy 2.54.
- Featured 50/50 from no guarantee: mean 93.51, sd 43.13, quantiles 90/95/99% = 156/158/161, skew 0.01, entropy 4.54; from guarantee: 62.34/24.34/80/81/83. Repeated targets m = 2: mean 187.01, sd 60.99, q90 265, q95 300; m = 5: 467.53, 96.44, 593, 626; m = 10: 935.06, 136.39, 1111, 1160.
- Normal approximation: max abs CDF error for T₁₀ ≈ 0.03.

If a reproduced value disagrees with the paper, the discrepancy is investigated and documented rather than forced (the paper may contain typos; q95 = 300 for m = 2 is a candidate).

Endfield model:

- Closed-form sanity with soft pity, guarantee, vacuum, dossier, potential and free pulls disabled: first-UP waiting time has E = E[cycle]/q with E[cycle] = Σ_{j=0}^{79} (1−p)^j; the kernel must match within 1e−9.
- Support bounds for target 1 copy: with f = 0, P(T ≤ 120) = 1 and P(T = 120) > 0; with f = 15, P(T ≤ 105) = 1.
- Entering pity t₀ = 79 with f = 0: P(T = 1) = 0.5.
- Guarantee void: starting with u = 1 and n = 119, the 120th pull is not forced.
- Vacuum: with target 1 copy the transition into n = 30 carries success mass 1 − (1 − vacuum_rate·q)^10 times the non-success branch mass; with vacuum disabled that mass is absent.
- Potential: with cap ≥ 240 and target 2 copies, reaching n = 240 without a second UP still absorbs success.
- Monotonicity in entering pity: E[T | t₀] is expected to be non-increasing in t₀ (checked over all t₀). This is not obvious: an earlier 6★ reaches t = 0 at a smaller n, so the 120 guarantee is further away in paid pulls. If the test fails, the failure is a research finding to document in `docs/research-notes.md`, not a bug to silence.
- Golden values for the default rules (first UP, fresh state, f = 5, cap 120): mean, sd, quantiles, P(T = 115) frozen after first computation with a comment linking to `docs/results.md`.

Plan:

- A one-banner plan equals `SingleBannerModel`.
- A skipped banner with cap 0 uses only free pulls: the next banner's entering pity distribution equals the 5-pull propagation of t₀.
- Dossier propagation: a wanted banner with cap ≥ 60 sets d = 1 for the next banner on the paths where n ≥ 60; the next banner then has f = 15. Verified by comparing against a hand-built two-banner chain and by Monte Carlo parity.
- Cross-banner coupling: the exact two-banner distribution differs from the i.i.d. convolution of single-banner PMFs (test asserts a nonzero max CDF difference and that exact P(success) ≥ convolution-based value when entering pity helps).

Monte Carlo parity (`test_mc_parity.py`, seed fixed, 200 000 paths, 3 standard errors): paper single counter (mean, P(T ≥ 80)); Endfield first UP (mean, P(T ≥ 60), P(success)); a two-banner plan (P(success), mean).

Performance (informal, asserted loosely in tests): single banner < 0.1 s; three-banner plan with caps 200 < 5 s; 200 000 Monte Carlo paths < 30 s.

## 7. Experiments (Phase 1 deliverables)

Each experiment module exposes `run(out_dir: Path) -> None`, writes CSV and Markdown tables to `results/tables/` and PNG + PDF figures to `results/figures/`, and appends headline numbers to `docs/results.md` sections it owns.

- E1 `reproduce_paper`: Figures 1–2 (schedules), Table 1 (three baselines), single-stage PMF/CDF, T₁₀ PMF, quantile curve, tail function, VaR/CVaR curve, featured 50/50 PMF comparison. Validation of the engine.
- E2 `endfield_single`: PMF and CDF of paid pulls to the first UP from a fresh state, three overlaid variants: (i) 80 pity + 50/50 only, (ii) + 120 guarantee, (iii) + vacuum, dossier-free start, free 5. Highlights the mass spike at 120 − f. Also 2 copies (potential + 240).
- E3 `personal_state`: heatmaps over (t, n) of E[remaining paid pulls to first UP] and P(first UP within b) for b ∈ {30, 60, 90}; same for entering t₀ only (line plot: "the value of pity").
- E4 `carry_over`: total paid pulls to obtain the UP on 2 and 3 consecutive wanted banners: exact plan vs the paper's i.i.d. convolution; report max CDF gap, quantile shifts, and P(success | budget) differences. Headline result.
- E5 `tail_risk`: for 1–5 consecutive targets: quantiles, VaR/CVaR, expected excess, completion probability vs budget (table + completion heatmap like paper Fig. 14).
- E6 `mc_convergence`: Monte Carlo estimate of a tail probability vs replications against the exact value (paper Fig. 11 style) for the Endfield first-UP model.
- E7 `sensitivity`: single-banner mean and 90% quantile under up_share ∈ {0.5, 0.6, 0.7}, soft_pity_start ∈ {60, 65, 70}, guarantee_pull ∈ {100, 120, 140, none}.

Figure conventions follow the `dataviz` skill (consistent palette, both themes not required for static figures, axis labels with units).

## 8. Research notes to record (`docs/research-notes.md`)

Contributions relative to Hou et al.:

1. A two-counter state (t carried, n banner-local) with a one-time expiring guarantee is a different state-dependent asymmetry from the (t, g) 50/50 model: the guarantee truncates support within a banner but not across banners.
2. Backward recurrences on a DAG with cost-0 and cost-1 transitions generalize Propositions 1–2; free pulls make T a weighted hitting time.
3. Cross-banner coupling through t and the dossier invalidates the i.i.d. convolution (paper Eq. 2, Remark 1); the exact state-carrying composition replaces it, and the gap is quantified (E4).
4. Propositions to derive: stochastic ordering in entering pity t₀ (monotone value of pity); stochastic ordering under the guarantee (guarantee_pull smaller ⇒ smaller T in the usual order); the vacuum bonus is an independent Bernoulli mixture; support and mass-spike characterization at 120 − f.
5. Phase 2 preview: chance-constrained DP over the same state space with per-banner caps as decisions; monetary risk via the purchase menu's piecewise cost curve; Phase 3: weapon banner and quota currencies.

## 9. Open questions for the user (non-blocking; defaults stated in §3)

1. Are the 5 free pulls per banner banner-bound permits or generic permits usable on any banner (R9)?
2. Do vacuum pulls use the flat base rate (R6)?
3. Do dossier and free-start pulls count toward n for the 30/60/120/240 milestones (R7, R9)?
4. Does the guaranteed 120th pull reset the pity counter (R5)?
5. Later phases: weapon banner milestones at 40 and 100 pulls, and the off-rate composition (R10).

## 10. Definition of done for Phase 1

- `uv run pytest` green, including paper reproduction and Monte Carlo parity.
- `uv run gacha experiment all` produces every table and figure in §7 without error.
- `docs/assumptions.md`, `docs/research-notes.md`, `docs/results.md`, `CLAUDE.md`, and the two skills exist and match the code.
