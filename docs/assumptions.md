# Rule and policy assumptions register

Status: **confirmed** = user statement and endfield.wiki.gg agree; **assumed** = modeling choice to verify.
Every row maps to a parameter of `EndfieldCharacterRules` (`src/gacha/rules/endfield.py`) or to a
policy in `src/gacha/models/endfield.py` / `plan.py`. Change a rule here first, then in code (see the
`add-mechanic` skill).

| ID | Rule | Status | Parameter / code |
|----|------|--------|------------------|
| R1 | 500 合成玉 per pull; 1 源石 = 75 合成玉 | confirmed | Phase 2 cost layer |
| R2 | p_t = 0.008 (t ≤ 64); 0.008 + 0.05·(t−64) (65 ≤ t ≤ 78); p_79 = 1. Applies to limited and re-run banners | confirmed: user 2026-10-07 (after briefly stating there was no soft pity the same day) and endfield.wiki.gg agree | base_rate, soft_pity_start, soft_pity_step, hard_pity |
| R3 | any 6★ from a counted pull resets t; t carries across limited banners | confirmed | `pull()` reset, `Plan` boundary |
| R4 | each counted 6★ is the UP w.p. 0.5, independently | confirmed | up_share |
| R5 | 120th counted pull of a banner is the UP if none obtained yet; one-time; resets t | confirmed (reset confirmed by user 2026-10-07) | guarantee_pull |
| R6 | at n = 30: 10 vacuum pulls, flat 0.008, no effect on t/n/u, copies add to c | confirmed, including the flat rate (user 2026-10-07) | vacuum_at, vacuum_pulls, vacuum_rate |
| R7 | n ≥ 60 at banner end → 10 free counted pulls at the start of the next banner | confirmed, including counting in n (user 2026-10-07) | dossier_at, dossier_pulls |
| R8 | every 240 counted pulls: +1 UP token (raises potential); an operator maxes out at 6 copies (1 + 5 potentials), so target_copies above 6 is meaningless | confirmed (cap of 6 from user 2026-10-07; not enforced in code) | potential_every |
| R9 | 10 banner-bound pulls per limited banner: 5 from logging in during the first three days of the version plus 5 exchanged in the shop (user 2026-10-07, "通常都會換"). The code default is still 5; use `free_start_pulls=10` for a player who takes the shop exchange. Not known for re-run banners (RR7 assumes 0) | confirmed banner-bound (user 2026-10-07) | free_start_pulls |
| R10 | off-rate 6★ composition (standard vs previous limited) | deferred (Phase 3) | — |
| R11 | 5★ layer: 8% base; if 9 pulls in a row give no 5★ or better, the 10th is 5★ or better (6★ at its pity rate, otherwise 5★). When the 6★ rate is boosted the 5★ rate is min(8%, 1 − p_t) | wiki; the boosted-rate interaction is assumed. Used only for the quota rebate | five_star_rate, five_star_pity |
| R12 | limited operator stays 3 banners | deferred (affects R10) | — |

| ID | Policy assumption | Code |
|----|-------------------|------|
| P1 | free pulls are used before paid pulls; paid = max(0, n − f) | `pull()` cost rule |
| P2 | free pulls are always used, even on skipped banners | `Plan` (cap 0 banners), `status()` |
| P3 | pull until target or cap; finish remaining free pulls after success | `status()` |
| P4 | T counts paid pulls only | `Transition.cost` |

## Re-run banner (重構尋訪), source: docs/rules/rerun-banner.md

Built with `rerun_rules()` in `src/gacha/rules/endfield.py`; cross-session persistence is expressed by
the caller passing the saved (t, n, c, u) as the start state.

| ID | Rule | Status | Parameter / code |
|----|------|--------|------------------|
| RR1 | independent series: no pity shared with the limited banner; coupled only through the pull stock | guide 2026-10-07 | separate chains, convolved |
| RR2 | the 80 pity counter t is shared by **all** re-run banners and carries between them; the banner-local counters n (30/60/90/120/240) persist only across same-named re-runs | t: official in-game rule text (screenshot 2026-10-07); n: guide | start state |
| RR3 | 10 uncounted bonus pulls at n = 30, 60 and 90, each once ever, flat 0.008, UP from them does not void the 120 guarantee | thresholds from guide; rate and isolation assumed (same as R6) | vacuum_at=(30, 60, 90) |
| RR4 | 80 hard pity with the same soft-pity ramp as R2, 50% UP | guide; ramp confirmed by user 2026-10-07 | as R2, R4 |
| RR5 | first 120 pulls guarantee the UP, void if obtained earlier | guide | guarantee_pull=120 |
| RR6 | 240 pulls: UP token | guide; "every 240" assumed | potential_every=240 |
| RR7 | no next-banner dossier and no free start pulls | assumed (guide silent) | dossier_at=None, free_start_pulls=0 |

### Calibration against the official published rates (2026-10-07)

The in-game rule text for 「絢麗異彩」 states a comprehensive 6★ rate (tokens included) of 2.0387% up to
and including the first UP, and 2.2720% afterwards. The model reproduces both to four decimals:
1/E[pity cycle] + 1/240 = 1/53.8993 + 1/240 = 2.2720%, and E[6★ until first UP] / E[counted pulls
until first UP] = 2.0387% (bonus pulls excluded). This confirms R2 (ramp from the 66th pull), R4,
R5 (120 guarantee) and R8/RR6 (one token per 240 pulls).

### 保障配額 rebate (R13), modeled as an expected-value approximation

25 保障配額 buy one universal permit; a duplicate 5★ gives 10 and a duplicate 6★ gives 50
(`quota_per_permit`, `quota_five_dupe`, `quota_six_dupe`). `gacha.analysis.rebate` computes the
long-run 5★ and 6★ rates from the stationary law of (6★ pity, 5★ pity) and turns a player's
ownership shares into quota per pull; the stock is then scaled by 1 / (1 − quota per pull / 25).
For a player whose 5★ are all owned this is about 1.3 quota per pull, roughly one extra pull per
19 pulls. It is an average, not part of the exact chain: the variance of the rebate, quota earned
from free and bonus pulls, and the start-up transient are ignored. The roster used by the site is
in `docs/rules/roster.md`.

## Open questions (spec §9)

1. Do the 5 free-start pulls count toward n (R9)? (dossier pulls do, R7)
2. Weapon banner milestones at 40/100 and off-rate composition (Phase 3).
