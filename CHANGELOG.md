# Changelog

## 0.2.0 — 2026-10-08

Added
- `gacha decide`: two banners with separate pity sharing one stock, from a TOML scenario file.
- `gacha weapon` and the weapon banner model (issues to the first rate-up weapon); arsenal quota
  earned from character pulls.
- Re-run banner rules (`rerun_rules()`): bonus pulls at 30/60/90 and persistent counters.
- `gacha.cost`: price menus and the cheapest top-up for a number of pulls.
- `gacha.analysis.rebate`: 保障配額 rebate as an expected-value layer, from the stationary 5★ and
  6★ rates.
- Documentation site (`docs/site`): calculator for "re-run plus next limited" and for three
  consecutive limited banners with skip patterns, weapons, a report generated from the inputs
  with money risk metrics and derivations. Its JavaScript engines are checked against the Python
  engine on every build.
- Data records: operator roster, past banner schedule, Taiwan price list, weapon and re-run rules.

Changed
- Rule register extended (R11, R13, RR1-RR7, W1-W6) and calibrated against the official
  comprehensive rates (2.0387% and 2.2720%, reproduced to four decimals).
- The CLI reports invalid input as a one-line error; kernel invariants are tested on every model.

Known limits
- The rebate is an average, not part of the exact chain. 集成配額, the 100/180-pull weapon
  milestones and adaptive stopping are not modelled. Multi-banner plans always use free pulls on
  skipped banners, which is not the best play near pity (docs/research-notes.md).

## 0.1.0 — 2026-09-23

- Absorbing-chain kernel: state enumeration, forward distribution over own pulls, backward
  expectation, variance and success-within-budget.
- Paper models and reproduction of Hou, Zhu and Zhang (2026).
- Endfield limited character banner and multi-banner plans with carried pity.
- Risk metrics, Monte Carlo parity checks, seven reproducible experiments, `gacha evaluate`.
