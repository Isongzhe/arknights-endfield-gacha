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

- 2026-09-23, paper reproduction: every value in `tests/test_paper_reproduction.py` matched the
  paper within tolerance on the first run, including the m = 2 repeated-featured 95% quantile of
  300 that looked suspicious in the spec review.
- 2026-09-23, monotonicity: E[T | t0] is non-increasing in t0 for the default rules
  (`test_expected_pulls_monotone_in_entering_pity` passes; E3 reports `monotone_in_t0 = 1`).
- 2026-09-23, carry-over (E4): for two consecutive wanted banners the i.i.d. convolution
  overstates the mean by 7.7 paid pulls (148.7 vs 140.9) and the maximum CDF gap is 0.20; for
  three banners 15.4 pulls (223.0 vs 207.6) and 0.15.
