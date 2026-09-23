"""E5: tail-risk summaries for 1-5 consecutive wanted banners and a completion heatmap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.plan import Plan
from gacha.plots.heatmaps import heatmap
from gacha.plots.style import apply_style, new_figure, save
from gacha.risk import metrics as rk
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules

from ._io import ensure_dirs, write_table

KS = (1, 2, 3, 4, 5)
ALPHAS = (0.5, 0.75, 0.9, 0.95, 0.99)


def run(out_dir: Path) -> dict[str, float]:
    apply_style()
    _, figures = ensure_dirs(out_dir)
    spec = BannerSpec(EndfieldCharacterRules(), 1, 120)
    budgets = np.arange(0, 5 * 115 + 1, 5)
    rows, completion = [], []
    hts = {}
    for k in KS:
        ht = hitting_time(EnumeratedChain.from_model(Plan([spec] * k)))
        hts[k] = ht
        row = {"banners": k, "mean": rk.mean(ht), "sd": rk.sd(ht)}
        row.update({f"q{round(a * 100)}": rk.quantile(ht, a) for a in ALPHAS})
        row.update({"cvar90": rk.cvar(ht, 0.9), "cvar95": rk.cvar(ht, 0.95)})
        for per in (60, 90):
            row[f"excess_at_{per}_per_banner"] = rk.expected_excess(ht, per * k)
            row[f"completion_{per}_per_banner"] = rk.completion(ht, per * k)
        rows.append(row)
        completion.append([rk.completion(ht, int(b)) for b in budgets])
    write_table(pd.DataFrame(rows), out_dir, "e05_tail_risk")

    fig, ax = new_figure(6.5, 3.6)
    heatmap(
        ax,
        np.array(completion),
        x=budgets,
        y=np.array(KS),
        xlabel="budget in paid pulls",
        ylabel="consecutive wanted banners",
        cbar_label="P(all UPs within budget)",
        vmin=0,
        vmax=1,
    )
    ax.set_yticks(list(KS))
    save(fig, figures, "e05_fig_completion")

    levels = np.linspace(0.7, 0.995, 60)
    fig, ax = new_figure()
    ax.plot(levels, [rk.quantile(hts[3], a) for a in levels], label="VaR")
    ax.plot(levels, [rk.cvar(hts[3], a) for a in levels], label="CVaR")
    ax.set_xlabel("risk level α")
    ax.set_ylabel("paid pulls")
    ax.set_title("Three consecutive wanted banners")
    ax.legend()
    save(fig, figures, "e05_fig_var_cvar_K3")

    by_k = {int(r["banners"]): r for r in rows}
    return {
        "q90_K1": by_k[1]["q90"],
        "q90_K3": by_k[3]["q90"],
        "q90_K5": by_k[5]["q90"],
        "mean_K5": by_k[5]["mean"],
        "cvar95_K5": by_k[5]["cvar95"],
        "completion_60_per_banner_K5": by_k[5]["completion_60_per_banner"],
    }
