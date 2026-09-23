# Rule and policy assumptions register

Status: **confirmed** = user statement and endfield.wiki.gg agree; **assumed** = modeling choice to verify.
Every row maps to a parameter of `EndfieldCharacterRules` (`src/gacha/rules/endfield.py`) or to a
policy in `src/gacha/models/endfield.py` / `plan.py`. Change a rule here first, then in code (see the
`add-mechanic` skill).

| ID | Rule | Status | Parameter / code |
|----|------|--------|------------------|
| R1 | 500 合成玉 per pull; 1 源石 = 75 合成玉 | confirmed | Phase 2 cost layer |
| R2 | p_t = 0.008 (t ≤ 64); 0.008 + 0.05·(t−64) (65 ≤ t ≤ 78); p_79 = 1 | confirmed | base_rate, soft_pity_start, soft_pity_step, hard_pity |
| R3 | any 6★ from a counted pull resets t; t carries across limited banners | confirmed | `pull()` reset, `Plan` boundary |
| R4 | each counted 6★ is the UP w.p. 0.5, independently | confirmed | up_share |
| R5 | 120th counted pull of a banner is the UP if none obtained yet; one-time; resets t | confirmed (reset assumed) | guarantee_pull |
| R6 | at n = 30: 10 vacuum pulls, flat 0.008, no effect on t/n/u, copies add to c | isolation confirmed; rate assumed | vacuum_at, vacuum_pulls, vacuum_rate |
| R7 | n ≥ 60 at banner end → 10 free counted pulls at the start of the next banner | confirmed; counted in n assumed | dossier_at, dossier_pulls |
| R8 | every 240 counted pulls: +1 UP copy | confirmed | potential_every |
| R9 | 5 free banner-bound pulls at banner start | user; banner-bound assumed | free_start_pulls |
| R10 | off-rate 6★ composition (standard vs previous limited) | deferred (Phase 3) | — |
| R11 | 5★ layer (8%, one per 10 pulls) | deferred (Phase 3) | — |
| R12 | limited operator stays 3 banners | deferred (affects R10) | — |

| ID | Policy assumption | Code |
|----|-------------------|------|
| P1 | free pulls are used before paid pulls; paid = max(0, n − f) | `pull()` cost rule |
| P2 | free pulls are always used, even on skipped banners | `Plan` (cap 0 banners), `status()` |
| P3 | pull until target or cap; finish remaining free pulls after success | `status()` |
| P4 | T counts paid pulls only | `Transition.cost` |

## Open questions (spec §9)

1. Are the 5 free pulls banner-bound or generic permits (R9)?
2. Vacuum pull rate flat 0.008 (R6)?
3. Do dossier and free-start pulls count toward n (R7, R9)?
4. Does the guaranteed 120th pull reset t (R5)?
5. Weapon banner milestones at 40/100 and off-rate composition (Phase 3).
