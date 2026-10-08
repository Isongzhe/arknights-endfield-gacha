# Documentation

Start with [../ARCHITECTURE.md](../ARCHITECTURE.md) for how the code is organised and [../ROADMAP.md](../ROADMAP.md) for what comes next.

| File | What it is |
|---|---|
| [assumptions.md](assumptions.md) | Every game rule the models use, its source and status, and the calibration against the official published rates |
| [research-notes.md](research-notes.md) | Contributions relative to the reference paper, propositions to prove, planned work, dated findings |
| [results.md](results.md) | Headline numbers written by `uv run gacha-risk experiment all` |
| [rules/README.md](rules/README.md) | Overview of every banner type with a side-by-side comparison (Chinese) |
| [rules/special-banner.md](rules/special-banner.md) | Special banner 輝光慶典 rules, recorded only (Chinese) |
| [rules/rerun-banner.md](rules/rerun-banner.md) | Re-run banner (重構尋訪) rules, schedule and the official rate text (Chinese) |
| [rules/weapon-banner.md](rules/weapon-banner.md) | Weapon banner rules and first calculations (Chinese) |
| [rules/banner-schedule.md](rules/banner-schedule.md) | Past character banners and versions, with what the schedule implies (Chinese) |
| [rules/roster.md](rules/roster.md) | Operator roster and the off-rate 6★ pool (Chinese) |
| [rules/pricing.md](rules/pricing.md) | Taiwan price list and purchase limits (Chinese) |
| [site/](site/) | Browsable write-up with calculators; `uv run python docs/site/build.py` builds `index.html` and `standalone.html` |
| [paper-outline.md](paper-outline.md) | Working outline of the paper: claims, status of each contribution, figures still to produce |

Data files used by the site live in `site/`: `roster.json`, `banners.json`, and the generated
`data.json`. Operator icons are game art and are not committed; `site/fetch_icons.py` downloads
them locally.
