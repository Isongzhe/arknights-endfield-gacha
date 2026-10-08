"""E3: value of the current state (t, n): expected remaining paid pulls and success within b."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gacharisk.kernel.backward import state_values, success_within
from gacharisk.kernel.chain import EnumeratedChain
from gacharisk.models.endfield import BannerState, SingleBannerModel
from gacharisk.plots.heatmaps import heatmap
from gacharisk.plots.style import apply_style, new_figure, save
from gacharisk.rules.endfield import BannerSpec, EndfieldCharacterRules

from ._io import ensure_dirs, write_table

BUDGETS = (30, 60, 90)


def run(out_dir: Path) -> dict[str, float]:
    apply_style()
    _, figures = ensure_dirs(out_dir)
    rules = EndfieldCharacterRules(free_start_pulls=0)  # n == paid pulls already made
    spec = BannerSpec(rules, 1, 120)
    nt, nn = rules.hard_pity, 120
    roots = [BannerState(t, n, 0, 0) for t in range(nt) for n in range(nn)]
    chain = EnumeratedChain.from_model(SingleBannerModel(spec), roots=roots)
    sv = state_values(chain)
    s = success_within(chain, max(BUDGETS))

    def grid(values: np.ndarray) -> np.ndarray:
        return np.array(
            [[values[chain.index[BannerState(t, n, 0, 0)]] for n in range(nn)] for t in range(nt)]
        )

    e_grid = grid(sv.expectation)
    p_grids = {b: grid(s[b]) for b in BUDGETS}

    fig, ax = new_figure(6.5, 4.2)
    heatmap(
        ax,
        e_grid,
        x=np.arange(nn),
        y=np.arange(nt),
        xlabel="pulls already made on this banner n",
        ylabel="pity counter t",
        cbar_label="expected remaining paid pulls to first UP",
    )
    ax.set_title("Expected remaining paid pulls (u = 0, no copies yet)")
    save(fig, figures, "e03_fig_expected_remaining")

    fig, axes = new_figure(11, 3.6, ncols=3)
    for ax, b in zip(axes, BUDGETS, strict=True):
        heatmap(
            ax,
            p_grids[b],
            x=np.arange(nn),
            y=np.arange(nt),
            xlabel="pulls already made n",
            ylabel="pity counter t",
            cbar_label=f"P(first UP within {b} paid pulls)",
            vmin=0,
            vmax=1,
        )
    save(fig, figures, "e03_fig_success_within")

    full = EndfieldCharacterRules()
    roots5 = [BannerState(t, 0, 0, 0) for t in range(nt)]
    chain5 = EnumeratedChain.from_model(SingleBannerModel(BannerSpec(full, 1, 120)), roots=roots5)
    sv5 = state_values(chain5)
    e_t0 = [float(sv5.expectation[chain5.index[r]]) for r in roots5]
    write_table(
        pd.DataFrame({"t0": range(nt), "expected_paid_pulls": e_t0}), out_dir, "e03_value_of_pity"
    )
    fig, ax = new_figure()
    ax.plot(range(nt), e_t0)
    ax.set_xlabel("entering pity t0")
    ax.set_ylabel("expected paid pulls to first UP")
    ax.set_title("The value of pity (default rules, f = 5)")
    save(fig, figures, "e03_fig_value_of_pity")

    selected = [(0, 0), (0, 60), (0, 100), (40, 40), (64, 0), (70, 0), (79, 0), (0, 119)]
    write_table(
        pd.DataFrame(
            [
                {
                    "t": t,
                    "n": n,
                    "expected_remaining": e_grid[t, n],
                    "p_within_30": p_grids[30][t, n],
                    "p_within_60": p_grids[60][t, n],
                    "p_within_90": p_grids[90][t, n],
                }
                for t, n in selected
            ]
        ),
        out_dir,
        "e03_selected_states",
    )
    return {
        "E_fresh": float(e_grid[0, 0]),
        "E_t79_n0": float(e_grid[79, 0]),
        "E_t0_n100": float(e_grid[0, 100]),
        "monotone_in_t0": float(bool(np.all(np.diff(e_t0) <= 1e-9))),
    }
