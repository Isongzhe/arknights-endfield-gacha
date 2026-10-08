# Documentation

| File | What it is |
|---|---|
| [assumptions.md](assumptions.md) | Every game rule the models use, its source and status, and the calibration against the official published rates |
| [research-notes.md](research-notes.md) | Contributions relative to the reference paper, propositions to prove, planned work, dated findings |
| [results.md](results.md) | Headline numbers written by `uv run gacha experiment all` |
| [rules/rerun-banner.md](rules/rerun-banner.md) | Re-run banner (重構尋訪) rules, schedule and the official rate text (Chinese) |
| [rules/weapon-banner.md](rules/weapon-banner.md) | Weapon banner rules and first calculations (Chinese) |
| [rules/banner-schedule.md](rules/banner-schedule.md) | Past character banners and versions, with what the schedule implies (Chinese) |
| [rules/roster.md](rules/roster.md) | Operator roster and the off-rate 6★ pool (Chinese) |
| [rules/pricing.md](rules/pricing.md) | Taiwan price list and purchase limits (Chinese) |
| [site/](site/) | Browsable write-up with calculators; `uv run python docs/site/build.py` builds `index.html` and `standalone.html` |
| [design/](design/) | The original Phase 1 design and implementation plan, kept as a record |

Data files used by the site live in `site/`: `roster.json`, `banners.json`, and the generated
`data.json`. Operator icons are game art and are not committed; `site/fetch_icons.py` downloads
them locally.
