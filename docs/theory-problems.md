# Theory problems

A list of the mathematical problems behind this project, written so that each can be studied,
cited or attacked on its own. For every problem: the statement, what is already established here
(and where), and what is open. Classical prototypes and references are collected at the end.

中文摘要：這份文件把專案背後的數學問題逐題寫成正式敘述，標明哪些已經解決、哪些還是開放問題，並列出對應的經典理論與參考文獻。

## Notation

| Symbol | Meaning |
|---|---|
| \(t\) | pity counter: pulls since the last 6★ |
| \(p_t\) | probability that the next pull is a 6★ at counter \(t\); \(p_{H-1}=1\) (hard pity at \(H\)) |
| \(q\) | probability that a 6★ is the rate-up target |
| \(G\) | banner-local guarantee: the \(G\)-th counted pull yields the target if it has not appeared |
| \(f\) | free pulls at the start of a banner: they advance all counters and cost nothing |
| \(T\) | number of own (paid) pulls until the target is obtained |
| \(S\) | stock: own pulls available |

Endfield limited banner: \(H=80\), \(p_t=0.008\) for \(t\le 64\), \(p_t=0.008+0.05\,(t-64)\) for
\(65\le t\le 78\), \(q=0.5\), \(G=120\). Rules and their sources: [assumptions.md](assumptions.md).

## P1. Waiting-time distribution of one banner — solved

**Statement.** Given \((p_t)\), \(q\), \(G\), \(f\) and an initial state, compute \(P(T=k)\) for all \(k\).

**Status.** Solved exactly. The state \((t,n,c,u)\) evolves on a directed acyclic graph because the
banner-local count \(n\) increases on every pull. \(T\) is a cost-weighted absorption time: free
pulls are cost-0 transitions. A forward recursion over cost levels gives the distribution in
\(O(G\cdot H)\) operations; backward recurrences give mean, variance and success-within-budget for
every state. With one counter and no guarantee this reduces to the recurrences of Hou, Zhu and
Zhang (2026). \(T\) is a discrete phase-type distribution.

**Where.** `gacharisk.kernel`, `gacharisk.models.endfield`; experiment E2.

## P2. Long-run rate from the pity cycle — solved

**Statement.** Show that the long-run frequency of 6★ results is \(1/E[C]\), where \(C\) is the
length of one pity cycle, \(E[C]=\sum_{j=0}^{H-1}\prod_{r<j}(1-p_r)\); and that periodic rewards
every \(m\) pulls add \(1/m\).

**Status.** Solved; this is the elementary renewal theorem and the renewal-reward theorem. For
Endfield \(E[C]=53.8993\), so \(1/E[C]+1/240=2.2720\%\), the operator's published figure after the
first target. The figure before the first target, \(2.0387\%\), equals the expected number of 6★
results divided by the expected number of counted pulls until the first target.

**Where.** [assumptions.md](assumptions.md), calibration section; `gacharisk.analysis.rebate.star_rates`.

## P3. Two banners, one stock — solved for cap policies

**Statement.** Two banners have independent waiting times \(T_A\), \(T_B\) with probability mass
functions \(f_A,f_B\) and distribution functions \(F_A,F_B\). The player spends at most \(x\) pulls
on \(A\), stops early on success, and spends the rest of the stock \(S\) on \(B\). Find
\(P_A(x)\), \(P_B(x)\), \(P_{\text{both}}(x)\) and describe how they change with \(x\).

**Status.** Solved for this policy class.

\[ P_{\text{both}}(x)=\sum_{j=0}^{x} f_A(j)\,F_B(S-j),\qquad P_A(x)=F_A(x),\qquad
   P_B(x)=P_{\text{both}}(x)+\bigl(1-F_A(x)\bigr)F_B(S-x). \]

\[ \Delta P_A=f_A(x),\qquad \Delta P_{\text{both}}=f_A(x)\,F_B(S-x)\ge 0,\qquad
   \Delta P_B=-\bigl(1-F_A(x-1)\bigr)\,f_B(S-x+1)\le 0 . \]

Consequences: \(P_{\text{both}}\) is non-decreasing in the cap; the gain of the \(x\)-th pull is
proportional to \(f_A(x)\) and the loss to \(f_B(S-x+1)\). Both mass functions are concentrated
at pity thresholds, so gains and losses are concentrated there and caps strictly between
thresholds are dominated by a neighbouring threshold.

**Open.** The same question for policies that react to what happens (see P5).

**Where.** `gacharisk.analysis.two_banner`; site report, derivation section.

## P4. Consecutive banners with carry-over — solved numerically

**Statement.** For \(K\) consecutive banners of the same series, the pity counter and a bonus earned
on banner \(k\) carry into banner \(k+1\). Compute the distribution of the total number of own
pulls and compare it with the \(K\)-fold convolution of the single-banner distribution.

**Status.** Solved exactly by composing banners with the carried state. The stages are dependent,
so the independent-sum formula is biased: for Endfield it overstates the mean by 7.7 pulls for two
banners (148.7 against 140.9) and by 15.4 for three, with a maximal CDF gap of 0.20 and 0.15.

**Open.** A bound on the gap in terms of the mechanism's parameters; conditions under which the
independent sum is exact or stochastically dominates the true total.

**Where.** `gacharisk.models.plan`; experiment E4.

## P5. Optimal adaptive policy under a budget — open

**Statement.** State: remaining stock, current banner, its counters, targets obtained. Action at
each step: pull on the current banner, or stop pulling on it. Objective: maximise the probability
of obtaining a given set of targets (or a weighted number of targets) with the stock.

**Status.** Open. It is a finite-horizon Markov decision process on the state space already
enumerated for P1 and P4, so the optimal value and policy are computable by backward induction in
\(O(S\cdot|\mathcal S|)\).

**Conjectures to prove or refute.**
1. An optimal policy stops only at pity thresholds, or exactly when the remaining stock equals
   what the next banner needs to reach its own threshold.
2. After an off-rate 6★ the optimal action depends on the state only through the distance to the
   guarantee and the remaining stock.
3. Concentrating the stock on one target is optimal when the stock cannot cover two guarantees
   (the "bold play" analogue).

**Prototype.** Maximising the probability of reaching a goal with limited resources is the
problem of Dubins and Savage; stochastic shortest path and constrained or risk-sensitive MDPs give
the computational framework.

## P6. Free pulls on a banner you do not want — numerical finding, open as a theorem

**Statement.** A skipped banner offers \(f\) free pulls. Using them advances the pity counter but
may release a 6★ there and reset it. Given the entering counter \(t_0\), should they be used?

**Status.** Numerically, with the next wanted banner valued and the stray 6★ valued at zero: using
them lowers the expected own pulls for the next target when \(t_0\) is small and raises it sharply
once \(t_0+f\) enters the soft-pity zone (about \(t_0+f\ge 67\) for Endfield). Example with ten
free pulls: \(t_0=48\): 46.3 pulls if unused, 41.6 if used; \(t_0=62\): 37.4 against 62.6.

**Open.** A proof of a threshold rule in \(t_0+f\), and the optimal number of free pulls to use
when it is strictly between 0 and \(f\).

**Where.** [research-notes.md](research-notes.md), 2026-10-08; site, three-banner mode.

## P7. Monotone value of pity — conjecture, verified numerically

**Statement.** Let \(E(t_0)\) be the expected own pulls to the target when the banner is entered at
counter \(t_0\). Show that \(E\) is non-increasing in \(t_0\).

**Status.** Holds for all 80 entering values under the default rules. It is not obvious: an early
6★ resets the counter at a small \(n\), further from the guarantee. A coupling argument is sketched
in [research-notes.md](research-notes.md).

**Open.** A complete proof, and the stronger claim of stochastic ordering of \(T\) in \(t_0\).

## P8. What published aggregate rates identify — partial

**Statement.** The operator publishes two aggregate rates. Which parameters of the mechanism are
determined by them, and which are not?

**Status.** Sensitivity is computed: shifting the start of the ramp by one pull, changing the ramp
step by one point, the target share by five points, or the guarantee by ten pulls each moves at
least one of the two figures in the third or fourth significant digit. The position of the hard
pity is practically unidentifiable: moving it from 80 to 90 changes neither figure to four
decimals, because the ramp almost always ends the cycle first.

**Open.** A formal identifiability statement; the smallest set of published aggregates that pins
down a given mechanism class. This bears on what probability-disclosure rules should require.

## P9. Mechanism design: mean cost against the mass of the worst case — open

**Statement.** Among mechanisms with a given expected cost per target, how small can the
probability of the worst case be, and what is the trade-off between mean, variance, maximum and
the conditional experience after losing the first 50/50?

**Status.** Open. Two data points from this project:

| | Endfield limited | 50/50 with carried guarantee (reference paper's schedule) |
|---|---|---|
| Mean | 74.3 | 93.5 |
| Maximum | 115 | 180 |
| Standard deviation | 36.0 | 43.1 |
| P(exactly the maximum) | 0.328 | about 0 |
| P(reach the guarantee given the first 6★ is off-rate) | 0.679 | n/a (the next 6★ is the target) |

Endfield is better in mean, spread and maximum, yet a third of players meet the exact worst case
on any banner. This is a candidate formalisation of "good on paper, bad in experience".

**Caveat (project owner, 2026-10-08).** The single-copy comparison does not explain why players
dislike the second mechanism at least as much as the first. Two things it leaves out: the usable
state of a character often needs several copies and a weapon, so the relevant cost is that of a
bundle; and an off-rate result is worth little to a player who wanted the current target. A
useful formulation therefore has to price the bundle and value off-target outcomes, not only the
first copy. For scale: six copies on one Endfield banner take 451.8 pulls on average (median 475,
90th percentile 593), helped by one token per 240 pulls; seven independent copies under the
reference schedule take 7 × 93.5 = 654.6 on average.

**Open.** Define the frontier for a bundle objective; characterise mechanisms on it; relate it to
risk measures such as CVaR, to the salience of a named worst-case event, and to the price of a
pull in each game.

## P10. Runs of worst-case outcomes — solved

**Statement.** Each banner ends at the guarantee with probability \(g\), independently. Find the
probability of at least one run of \(r\) such banners among \(N\).

**Status.** Solved by Markov chain embedding with states "current run length". For Endfield
\(g=0.328\): three in a row has probability 0.035; at least one run of three occurs with
probability 0.106 in six banners and 0.235 in twelve.

## Classical prototypes

| Topic in this project | Classical problem | Entry points |
|---|---|---|
| Waiting time to the target | Absorption times of finite Markov chains; phase-type distributions | Grinstead and Snell, ch. 11; Norris; Neuts; Bladt and Nielsen |
| Long-run rates | Renewal and renewal-reward theorems | Ross, renewal chapter; Asmussen |
| Rising probability with the counter | Increasing failure rate distributions, stochastic orders | Barlow and Proschan; Shaked and Shanthikumar |
| Reaching a goal with limited stock | Gambler's ruin; bold play | Dubins and Savage |
| When to stop, how to route effort | Optimal stopping; stochastic shortest path; MDPs | Puterman; Bertsekas and Tsitsiklis |
| Allocating effort among banners | Pandora's box; bandit indices | Weitzman; Gittins |
| Worst-case spend | VaR and CVaR; risk-sensitive MDPs | Rockafellar and Uryasev; Bäuerle and Ott; Chow et al. |
| Streaks of bad luck | Runs in Bernoulli trials | Feller; Fu and Koutras |

## References

Bibliographic details were written from memory and must be checked against the sources before
they are used in a publication.

- Asmussen, S. *Applied Probability and Queues*, 2nd ed. Springer, 2003.
- Barlow, R. E.; Proschan, F. *Mathematical Theory of Reliability*. Wiley, 1965.
- Bäuerle, N.; Ott, J. Markov decision processes with average-value-at-risk criteria.
  *Mathematical Methods of Operations Research* 74 (2011), 361–379.
- Bertsekas, D. P.; Tsitsiklis, J. N. An analysis of stochastic shortest path problems.
  *Mathematics of Operations Research* 16 (1991), 580–595.
- Bladt, M.; Nielsen, B. F. *Matrix-Exponential Distributions in Applied Probability*. Springer, 2017.
- Chow, Y.; Tamar, A.; Mannor, S.; Pavone, M. Risk-sensitive and robust decision-making: a CVaR
  optimization approach. *Advances in Neural Information Processing Systems* 28 (2015).
- Dubins, L. E.; Savage, L. J. *How to Gamble If You Must: Inequalities for Stochastic Processes*.
  McGraw-Hill, 1965.
- Feller, W. *An Introduction to Probability Theory and Its Applications*, vol. 1, 3rd ed. Wiley, 1968.
- Fu, J. C.; Koutras, M. V. Distribution theory of runs: a Markov chain approach.
  *Journal of the American Statistical Association* 89 (1994), 1050–1058.
- Gan, T. Gacha game: when prospect theory meets optimal pricing. arXiv:2208.03602, 2022.
- Gittins, J. C. Bandit processes and dynamic allocation indices.
  *Journal of the Royal Statistical Society B* 41 (1979), 148–177.
- Grinstead, C. M.; Snell, J. L. *Introduction to Probability*, 2nd rev. ed. American Mathematical
  Society, 1997.
- Hou, S.; Zhu, Y.; Zhang, S. State-dependent asymmetry in soft-pity gacha waiting-time models:
  exact recurrences, tail risk, and featured-target extensions. *Symmetry* 18(6) (2026), 1051.
  doi:10.3390/sym18061051.
- Neuts, M. F. *Matrix-Geometric Solutions in Stochastic Models*. Johns Hopkins University Press, 1981.
- Norris, J. R. *Markov Chains*. Cambridge University Press, 1997.
- Puterman, M. L. *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley, 1994.
- Rockafellar, R. T.; Uryasev, S. Optimization of conditional value-at-risk.
  *Journal of Risk* 2(3) (2000), 21–41.
- Ross, S. M. *Introduction to Probability Models*. Academic Press (any recent edition).
- Shaked, M.; Shanthikumar, J. G. *Stochastic Orders*. Springer, 2007.
- Weitzman, M. L. Optimal search for the best alternative. *Econometrica* 47 (1979), 641–654.
