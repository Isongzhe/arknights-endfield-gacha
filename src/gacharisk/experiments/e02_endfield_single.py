"""E2: paid pulls to the first UP on one Endfield banner, rule variants overlaid."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gacharisk.kernel.chain import EnumeratedChain
from gacharisk.kernel.forward import hitting_time
from gacharisk.models.endfield import SingleBannerModel
from gacharisk.plots.style import apply_style, new_figure, save
from gacharisk.plots.waiting_time import plot_cdf, plot_pmf
from gacharisk.risk import metrics as rk
from gacharisk.rules.endfield import BannerSpec, EndfieldCharacterRules

from ._io import ensure_dirs, write_table


def _ht(model):
    return hitting_time(EnumeratedChain.from_model(model))


def run(out_dir: Path) -> dict[str, float]:
    apply_style()
    _, figures = ensure_dirs(out_dir)
    bare = EndfieldCharacterRules(
        guarantee_pull=None,
        vacuum_at=None,
        dossier_at=None,
        potential_every=None,
        free_start_pulls=0,
    )
    full = EndfieldCharacterRules()
    variants = [
        ("80 pity + 50/50 only (cap 480)", SingleBannerModel(BannerSpec(bare, 1, 480))),
        (
            "+ 120 guarantee",
            SingleBannerModel(BannerSpec(replace(bare, guarantee_pull=120), 1, 120)),
        ),
        ("full rules, no dossier (f=5)", SingleBannerModel(BannerSpec(full, 1, 120))),
        (
            "full rules, with dossier (f=15)",
            SingleBannerModel(BannerSpec(full, 1, 120), dossier=True),
        ),
    ]
    rows, hts = [], []
    for name, model in variants:
        ht = _ht(model)
        hts.append((name, ht))
        rows.append(
            {
                "variant": name,
                "p_success": ht.p_success,
                "mean": rk.mean(ht),
                "sd": rk.sd(ht),
                "q50": rk.quantile(ht, 0.5),
                "q90": rk.quantile(ht, 0.9),
                "q95": rk.quantile(ht, 0.95),
                "q99": rk.quantile(ht, 0.99),
                "support_max": int(np.nonzero(ht.pmf_stop)[0].max()),
            }
        )
    write_table(pd.DataFrame(rows), out_dir, "e02_variants")

    two = _ht(SingleBannerModel(BannerSpec(full, 2, 300)))
    write_table(
        pd.DataFrame(
            [
                {
                    "target_copies": 2,
                    "cap": 300,
                    "p_success": two.p_success,
                    "mean": rk.mean(two),
                    "q90": rk.quantile(two, 0.9),
                    "q99": rk.quantile(two, 0.99),
                }
            ]
        ),
        out_dir,
        "e02_two_copies",
    )

    fig, ax = new_figure(6.5, 3.8)
    for name, ht in hts:
        plot_pmf(ax, ht.pmf_stop, label=name)
    ax.set_xlim(0, 200)
    ax.set_title("Paid pulls to the first UP: Endfield rule variants")
    ax.legend()
    save(fig, figures, "e02_fig_pmf")

    fig, ax = new_figure(6.5, 3.8)
    for name, ht in hts:
        plot_cdf(ax, ht.pmf_stop, label=name)
    ax.set_xlim(0, 200)
    ax.legend()
    save(fig, figures, "e02_fig_cdf")

    fig, ax = new_figure()
    plot_pmf(ax, two.pmf_stop)
    ax.set_title("Two UP copies on one banner (240-pull potential included)")
    save(fig, figures, "e02_fig_two_copies_pmf")

    full_ht = hts[2][1]
    return {
        "full_mean": rk.mean(full_ht),
        "full_sd": rk.sd(full_ht),
        "full_q90": rk.quantile(full_ht, 0.9),
        "full_support_max": int(np.nonzero(full_ht.pmf_stop)[0].max()),
        "two_copies_mean": rk.mean(two),
        "two_copies_p_success": two.p_success,
    }
