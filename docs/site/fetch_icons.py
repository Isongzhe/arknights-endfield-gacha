"""Download operator icons into docs/site/icons/ for the site's ownership checklist.

The icons are game art owned by the publisher. They are not part of this repository (the folder
and the built pages are git-ignored); fetch them yourself if you want them in your local build:

    uv run python docs/site/fetch_icons.py
"""
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
roster = json.loads((HERE / "roster.json").read_text())
names = [o["en"] for o in roster["five"] + roster["six_offrate"]] + roster["six_other"] + roster["four"]
(HERE / "icons").mkdir(exist_ok=True)
for name in names:
    file = name.replace(" ", "_")
    target = HERE / "icons" / f"{file}.png"
    if target.exists():
        continue
    req = urllib.request.Request(
        roster["icon_url"].format(file=file), headers={"User-Agent": "Mozilla/5.0 gacha-docs"}
    )
    try:
        target.write_bytes(urllib.request.urlopen(req, timeout=30).read())
        print("ok  ", name)
    except Exception as exc:  # noqa: BLE001
        print("fail", name, exc)
