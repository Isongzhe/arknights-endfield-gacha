# Changelog

## Unreleased

- `gacha decide`: two banners with separate pity sharing one stock, from a TOML scenario file.
- `gacha.cost`: price menus and the cheapest top-up for a number of pulls.
- Re-run banner rules (`rerun_rules()`): bonus pulls at 30/60/90 and persistent counters.
- Rule register updated with confirmed rules and a calibration against the official
  comprehensive rates (2.0387% and 2.2720%).
- Documentation site with an interactive calculator whose JavaScript engine is verified against
  the Python engine at build time.
- CLI reports invalid input as a one-line error instead of a traceback.
- Kernel invariants are now tested on every real model.

## 0.1.0 — 2026-09-23

- Absorbing-chain kernel: state enumeration, forward distribution over own pulls, backward
  expectation, variance and success-within-budget.
- Paper models and reproduction of Hou, Zhu and Zhang (2026).
- Endfield limited character banner and multi-banner plans with carried pity.
- Risk metrics, Monte Carlo parity checks, seven reproducible experiments, `gacha evaluate`.
