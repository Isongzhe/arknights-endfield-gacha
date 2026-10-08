"""Rebuild docs/site/index.html: recompute chart data with the exact engine and inject it
into template.html. Run: uv run python docs/site/build.py"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from gacharisk.kernel.backward import state_values
from gacharisk.kernel.chain import EnumeratedChain
from gacharisk.kernel.forward import hitting_time
from gacharisk.models.endfield import BannerState, SingleBannerModel
from gacharisk.models.plan import Plan
from gacharisk.risk import metrics as rk
from gacharisk.rules.endfield import BannerSpec, EndfieldCharacterRules
from gacharisk.rules.paper import PaperSchedule

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

    from gacharisk.rules.endfield import rerun_rules

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
    from gacharisk.models.weapon import WeaponBannerModel, WeaponState
    from gacharisk.rules.weapon import WeaponBannerRules

    wpayload = []
    for done in (0, 3, 4, 7):
        m = WeaponBannerModel(WeaponBannerRules(), start=WeaponState(done * 10, (done * 10) % 40))
        wpayload.append({"o": {"issuesDone": done}, "pmf": [float(x) for x in ht(m)[0].f_succ]})
    wout = subprocess.run(
        ["node", "-e", runner.replace("{firstUp}", "{weaponUp:firstUp}"), str(HERE / "engine.js")],
        input=json.dumps(wpayload), capture_output=True, text=True,
    )
    if wout.returncode != 0:
        raise SystemExit(f"browser weapon engine disagrees with Python: {wout.stdout} {wout.stderr}")
    print("browser weapon engine max abs diff vs Python:", wout.stdout.strip())
    from gacharisk.models.plan import Plan

    ppayload = []
    cases3 = ((0, 5, [1, 1, 1], 0), (48, 10, [1, 0, 1], 0), (70, 5, [0, 1, 1], 0), (30, 10, [1, 1, 0], 0), (48, 10, [1, 1, 0], 1), (20, 5, [0, 1, 1], 1))
    for t0, free, want, d0 in cases3:
        rules = replace(full, free_start_pulls=free)
        specs = [BannerSpec(rules, 1, 120) if w else BannerSpec(rules, 0, 0) for w in want]
        h, _ = ht(Plan(specs, start=BannerState(t0, 0, 0, 0), dossier0=bool(d0)))
        o = {"t0": t0, "free": free, "want": [bool(w) for w in want], "useFree": True, "d0": bool(d0)}
        ppayload.append({"o": o, "pmf": [float(x) for x in h.f_succ]})
    prunner = runner.replace("{firstUp}", "{limitedPlan}").replace(
        "const p=firstUp(c.o);", "const st=limitedPlan(c.o).stages;const p=st[st.length-1];"
    )
    pout = subprocess.run(
        ["node", "-e", prunner, str(HERE / "engine.js")],
        input=json.dumps(ppayload), capture_output=True, text=True,
    )
    if pout.returncode != 0:
        raise SystemExit(f"browser plan engine disagrees with Python: {pout.stdout} {pout.stderr}")
    print("browser plan engine max abs diff vs Python:", pout.stdout.strip())
    print("browser engine max abs diff vs Python:", out.stdout.strip())


check_browser_engine()
from gacharisk.analysis.rebate import star_rates  # noqa: E402

rate6, rate5 = star_rates(full)
d["rates"] = {"six": rate6, "five": rate5}
data = json.dumps(d, ensure_ascii=False, separators=(",", ":"))
(HERE / "data.json").write_text(data)
template = (HERE / "template.html").read_text()
assert template.count("__DATA__") == 1
assert template.count("__ENGINE__") == 1


def checklist(entries, cls):
    """Ownership checkboxes; an icon is embedded when docs/site/icons/<name>.png exists."""
    import base64

    out = []
    for i, entry in enumerate(entries):
        en = entry["en"]
        label = f"{entry['zh']} {en}" if entry.get("zh") else en
        icon = HERE / "icons" / f"{en.replace(' ', '_')}.png"
        img = ""
        if icon.exists():
            b64 = base64.b64encode(icon.read_bytes()).decode()
            img = f'<img alt="" src="data:image/png;base64,{b64}">'
        out.append(
            f'        <label class="check" for="in-{cls}-{i}"><input id="in-{cls}-{i}" '
            f'class="{cls}" type="checkbox" checked>{img}{label}</label>'
        )
    return "\n".join(out)


roster = json.loads((HERE / "roster.json").read_text())
template = template.replace("__OWN5__", checklist(roster["five"], "own5"))
template = template.replace("__OWN6__", checklist(roster["six_standard"], "own6"))
page = template.replace("__DATA__", data).replace("__ENGINE__", (HERE / "engine.js").read_text())
(HERE / "index.html").write_text(page)
# index.html is a page body for hosts that supply the document shell; standalone.html is a
# complete document for a plain web server or GitHub Pages.
SHELL = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{color-scheme:light}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>
</head><body>
__PAGE__
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad:true,theme:matchMedia('(prefers-color-scheme: dark)').matches?'dark':'default'});</script>
</body></html>
"""
(HERE / "standalone.html").write_text(SHELL.replace("__PAGE__", page))
print({x["name"]: x["states"] for x in d["pmf"]}, {k: v["states"] for k, v in d["carry"].items()})
print("e_t0 range", min(d["e_t0"]), max(d["e_t0"]), "pmf max", max(max(x["y"]) for x in d["pmf"]))
