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

## Planned

The open problems are stated formally in [theory-problems.md](theory-problems.md) (P5 to P9).

- **Hybrid 保障配額 rebate.** Put the 6★ part into the exact chain: an off-rate 6★ that the player
  already owns grants 50 quota, i.e. two pulls, so the state only needs the number of banked
  rebate pulls. This part is lumpy and correlated with the outcome (it arrives exactly when the
  50/50 is lost), which is where the current mean-field treatment is biased. Keep the 5★ part as
  an average: many small independent contributions of 0.4 pull. Until then the site reports the
  effect of the rebate being two pulls higher or lower.

## Findings

- 2026-10-08, several copies on one banner. Six copies (full potential): mean 451.8 own pulls,
  sd 108.8, maximum 1195. By counted pull 479 the probability is 0.489, by pull 480 it is 0.749:
  the second token arrives on pull 480 and that single pull carries 26% of the mass; pull 720
  brings it to 0.993. A renewal estimate (first copy 74.3, then 1 / (q/E[C] + 1/240) = 74.4 pulls
  per copy) gives 446, close to the exact mean, so averages do not need the chain. A normal
  approximation gives 0.583 and 0.586 at pulls 479 and 480 and misses the jump entirely, so
  questions of the form "is my stock enough" do need the exact distribution.
- 2026-10-08, rate-up copies from probability alone in a 240-pull window starting at pity 0:
  0: 0.072, 1: 0.257, 2: 0.350, 3: 0.227, 4: 0.076; mean 2.03. Reaching pull 240 with only the
  guarantee and the token has probability 0.341 × 0.339 = 0.115.
- 2026-10-08, why a banner can be good on paper and feel bad (theory-problems P9, P10). Against
  the reference paper's 50/50 schedule, Endfield has the lower mean (74.3 vs 93.5), spread and
  maximum (115 vs 180), but 32.8% of outcomes are exactly the maximum (about 0 there), 67.9% of
  first-6★ losses run to the guarantee, and a run of three guarantee outcomes has probability
  0.035 per three banners and 0.235 somewhere in twelve.

- 2026-10-08, free pulls on a skipped banner. Expected own pulls for the next wanted UP (10 free
  pulls there), by entering pity t0, when the skipped banner's free pulls are not used / 5 used /
  10 used: t0 = 0: 69.5 / 67.8 / 66.2; t0 = 48: 46.3 / 43.9 / 41.6; t0 = 58: 39.4 / 38.2 / 46.7;
  t0 = 62: 37.4 / 41.9 / 62.6; t0 = 70: 35.5 / 66.3 / 67.1. Far from pity the free pulls bank a
  few pulls of pity; once they reach the soft-pity zone they usually release the 6★ on the banner
  nobody wanted and the saved pity is gone. Policy P2 (always use them) is therefore suboptimal
  for t0 + free pulls above about 67, and the 集成配額 shop exchange should be skipped there too.
  The comparison values only the wanted UP; the stray 6★ is counted as worth nothing.

- 2026-10-07, rule R2: for a few hours the default schedule was flat 0.8% (the user believed
  Endfield had no soft pity), then the user confirmed the ramp does exist on both limited and
  re-run banners, matching endfield.wiki.gg. The default is again +5% per pull from the 66th pull
  and GOLDEN is back at its 2026-09-23 values. The flat schedule stays reachable with
  `soft_pity_step = 0`; E7 reports it: mean 77.79 vs 74.33, median 75 vs 67, same q90 of 115.
- 2026-10-07, re-run banner (重構尋訪) added as `rerun_rules()`: bonus pulls at 30/60/90 and
  counters that persist across same-named re-runs (docs/rules/rerun-banner.md, RR1-RR7).
- 2026-09-23, paper reproduction: every value in `tests/test_paper_reproduction.py` matched the
  paper within tolerance on the first run, including the m = 2 repeated-featured 95% quantile of
  300 that looked suspicious in the spec review.
- 2026-09-23, monotonicity: E[T | t0] is non-increasing in t0 for the default rules
  (`test_expected_pulls_monotone_in_entering_pity` passes; E3 reports `monotone_in_t0 = 1`).
- 2026-09-23, carry-over (E4): for two consecutive wanted banners the i.i.d. convolution
  overstates the mean by 7.7 paid pulls (148.7 vs 140.9) and the maximum CDF gap is 0.20; for
  three banners 15.4 pulls (223.0 vs 207.6) and 0.15.
