"""Rebuild docs/site/index.html: recompute chart data with the exact engine and inject it
into template.html. Run: uv run python docs/site/build.py"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from gacha.kernel.backward import state_values
from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.endfield import BannerState, SingleBannerModel
from gacha.models.plan import Plan
from gacha.risk import metrics as rk
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules
from gacha.rules.paper import PaperSchedule

HERE = Path(__file__).parent


def r(a, n=5):
    return [round(float(x), n) for x in a]


def ht(model):
    c = EnumeratedChain.from_model(model)
    return hitting_time(c), c.n


full = EndfieldCharacterRules()
bare = replace(
    full, guarantee_pull=None, vacuum_at=None, dossier_at=None, potential_every=None,
    free_start_pulls=0,
)
d = {"sched_endfield": r(full.probs, 4), "sched_paper": r(PaperSchedule().probs, 4), "pmf": []}
variants = [
    ("僅 80 保底 + 50/50", SingleBannerModel(BannerSpec(bare, 1, 480))),
    ("加上 120 保底", SingleBannerModel(BannerSpec(replace(bare, guarantee_pull=120), 1, 120))),
    ("完整規則（免費 5 抽）", SingleBannerModel(BannerSpec(full, 1, 120))),
    ("完整規則 + 上池贈 10 抽", SingleBannerModel(BannerSpec(full, 1, 120), dossier=True)),
]
for name, m in variants:
    h, n = ht(m)
    d["pmf"].append({"name": name, "y": r(h.pmf_stop[:201]), "states": n})
spec = BannerSpec(full, 1, 120)
roots = [BannerState(t, 0, 0, 0) for t in range(full.hard_pity)]
c = EnumeratedChain.from_model(SingleBannerModel(spec), roots=roots)
sv = state_values(c)
d["e_t0"] = r([sv.expectation[c.index[x]] for x in roots], 3)
single, _ = ht(SingleBannerModel(spec))
d["carry"] = {}
for k in (2, 3):
    ex, n = ht(Plan([spec] * k))
    iid = rk.convolve(single.pmf_stop, k)
    size = max(len(ex.pmf_stop), len(iid))
    ce = np.cumsum(np.pad(ex.pmf_stop, (0, size - len(ex.pmf_stop))))
    ci = np.cumsum(np.pad(iid, (0, size - len(iid))))
    d["carry"][k] = {"exact": r(ce, 4), "iid": r(ci, 4), "states": n}
budgets = list(range(0, 580, 5))
grid = []
for k in range(1, 6):
    h, _ = ht(Plan([spec] * k))
    grid.append(r([rk.completion(h, b) for b in budgets], 3))
d["completion"] = {"budgets": budgets, "grid": grid}


def check_browser_engine():
    """The page's JS port must reproduce the Python engine before the site is built."""
    import subprocess

    from gacha.rules.endfield import rerun_rules

    flat = replace(full, soft_pity_step=0.0)
    rerun = rerun_rules()
    cases = [
        (rerun, BannerState(30, 30, 0, 0), False),
        (full, BannerState(48, 0, 0, 0), False),
        (full, BannerState(0, 0, 0, 0), True),
        (flat, BannerState(10, 20, 0, 0), False),
        (rerun, BannerState(79, 95, 0, 0), False),
        (replace(rerun, soft_pity_step=0.0), BannerState(30, 30, 0, 0), False),
    ]
    payload = []
    for rules, start, dossier in cases:
        h, _ = ht(SingleBannerModel(BannerSpec(rules, 1, 120), start=start, dossier=dossier))
        o = {
            "t": start.t, "n": start.n, "free": rules.free_pulls(dossier), "guar": 120,
            "vac": list(rules.vacuum_points), "soft": rules.soft_pity_step > 0,
        }
        payload.append({"o": o, "pmf": [float(x) for x in h.f_succ]})
    runner = (
        "const {firstUp}=require(process.argv[1]);let worst=0;"
        "for(const c of JSON.parse(require('fs').readFileSync(0,'utf8'))){const p=firstUp(c.o);"
        "const n=Math.max(p.length,c.pmf.length);for(let i=0;i<n;i++)"
        "worst=Math.max(worst,Math.abs((p[i]||0)-(c.pmf[i]||0)));}"
        "console.log(worst);process.exit(worst<1e-9?0:1)"
    )
    out = subprocess.run(
        ["node", "-e", runner, str(HERE / "engine.js")],
        input=json.dumps(payload), capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"browser engine disagrees with Python: {out.stdout} {out.stderr}")
    print("browser engine max abs diff vs Python:", out.stdout.strip())


check_browser_engine()
data = json.dumps(d, ensure_ascii=False, separators=(",", ":"))
(HERE / "data.json").write_text(data)
template = (HERE / "template.html").read_text()
assert template.count("__DATA__") == 1
assert template.count("__ENGINE__") == 1
page = template.replace("__DATA__", data).replace("__ENGINE__", (HERE / "engine.js").read_text())
(HERE / "index.html").write_text(page)
print({x["name"]: x["states"] for x in d["pmf"]}, {k: v["states"] for k, v in d["carry"].items()})
print("e_t0 range", min(d["e_t0"]), max(d["e_t0"]), "pmf max", max(max(x["y"]) for x in d["pmf"]))
