# Paper outline (working draft)

Status: outline only. Sections marked **ready** have results and code; **partial** have numbers but
no proof or no general statement; **missing** are not started. Update this file as work lands.

## Working title

Exact waiting-time distributions and budget risk for multi-counter pity systems, with carry-over
between banners: the case of Arknights: Endfield

## One-paragraph claim

Existing waiting-time models for pity-based gacha treat successive targets as independent stages
of a single counter. Deployed systems combine several counters, one-time guarantees that expire,
free pulls, and carry-over between banners, which makes stages dependent. We give an exact
finite-state treatment of this class, validate the inferred mechanism against rates published by
the operator, quantify the error of the independence assumption, and show that under a budget the
value of an additional pull is concentrated at pity thresholds, so that sensible stopping points
lie on those thresholds and monetary risk is bimodal.

## Contributions

| # | Contribution | Status | Evidence in the repository |
|---|---|---|---|
| 1 | Model class: several counters, an expiring one-time guarantee, cost-0 pulls; the waiting time is a cost-weighted absorption time on a DAG | ready | `gacharisk.kernel`, `gacharisk.models.endfield`, `docs/assumptions.md` |
| 2 | Exact algorithms: level-closure forward recursion and backward recurrences that reduce to the reference paper's propositions | ready | `gacharisk.kernel.forward`, `gacharisk.kernel.backward`, reproduction tests |
| 3 | Calibration against published aggregate rates (2.0387%, 2.2720%) and its sensitivity to each rule | ready | `docs/assumptions.md`; sensitivity table still to be scripted as an experiment |
| 4 | Dependence between banners: error of i.i.d. convolution | ready | experiment E4 |
| 5 | Two banners sharing a stock: closed forms for the marginal effect of the cap, threshold structure | partial: formulas derived and checked numerically, no formal statement | `gacharisk.analysis.two_banner`, site report section 9 |
| 6 | Monetary risk: piecewise price menu, VaR/CVaR in currency, bimodality | partial: in the site and scripts, not yet in the package | `gacharisk.cost`, site report section 6 |
| 7 | Free pulls on a skipped banner are harmful near pity | partial: numerical finding | `docs/research-notes.md` (2026-10-08) |
| 8 | Optimal stopping under a budget as an MDP; structure of the optimal policy | missing | — |
| 9 | Generality: second mechanism or abstract class | missing | — |

## Section plan

1. Introduction: what single-counter models leave out; consumer-side risk as the motivation.
2. Related work: reference paper (Hou, Zhu, Zhang 2026); absorbing chains and hitting times;
   loot-box and probability-disclosure literature; gacha pricing (Gan 2022).
3. Mechanism and model class: counters, guarantees, free pulls, carry-over; state space; DAG.
4. Exact computation: forward and backward algorithms; complexity; reduction to the reference
   paper.
5. Validation: reproduction; calibration to published rates and what it can and cannot detect;
   Monte Carlo.
6. Results for one banner: distribution shape, value of pity, sensitivity.
7. Dependence across banners: exact composition versus independent sums.
8. Budgeted decisions: two banners and one stock; threshold structure; free pulls on skipped
   banners; (optimal policy, once available).
9. Monetary risk: price menu, risk measures, bimodality.
10. Limitations: unverified rules, expected-value rebate, single game, no behavioural claims.

## Figures and tables to produce

| Item | Source | Exists |
|---|---|---|
| Hazard schedule, reference paper vs Endfield | E1/E2 | yes |
| First-UP distribution under rule variants | E2 | yes |
| Value of pity (expected pulls by entering counter) | E3 | yes |
| Exact vs independent-sum CDF for 2 and 3 banners | E4 | yes |
| Tail-risk table for 1 to 5 banners | E5 | yes |
| Calibration sensitivity table (each rule perturbed) | new experiment | no |
| Cap on first banner vs P(first), P(second), P(both) | new experiment from `two_banner` | no |
| Marginal gain and loss per pull with threshold positions | new experiment | no |
| Spend distribution and VaR/CVaR by strategy | new experiment from `gacharisk.cost` | no |
| Free pulls on a skipped banner by entering pity | new experiment | no |

## What must not be claimed

- An "optimal strategy" without a stated objective; optimality depends on how a player values
  each target.
- Anything about player behaviour, harm or operator revenue.
- That the model describes the live servers: it checks that the published rules and the published
  rates agree with each other.

## Candidate venues

Open-access applied mathematics or game-studies venues comparable to the reference paper's;
an arXiv preprint (math.PR or cs.GT) first.
