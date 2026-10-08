"""E6: Monte Carlo estimates converge to the exact tail probability (paper Fig. 11 style)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gacharisk.kernel.chain import EnumeratedChain
from gacharisk.kernel.forward import hitting_time
from gacharisk.mc.compare import compare
from gacharisk.mc.simulate import simulate
from gacharisk.models.endfield import SingleBannerModel
from gacharisk.plots.style import PALETTE, apply_style, new_figure, save
from gacharisk.risk import metrics as rk
from gacharisk.rules.endfield import BannerSpec, EndfieldCharacterRules

from ._io import ensure_dirs, write_table

THRESHOLD = 90
NS = (100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 200_000)
SEED = 20260923


def run(out_dir: Path) -> dict[str, float]:
    apply_style()
    _, figures = ensure_dirs(out_dir)
    model = SingleBannerModel(BannerSpec(EndfieldCharacterRules(), 1, 120))
    chain = EnumeratedChain.from_model(model)
    ht = hitting_time(chain)
    exact = rk.survival(ht, THRESHOLD)
    paths = simulate(chain, NS[-1], seed=SEED)
    hits = np.cumsum((paths.paid >= THRESHOLD).astype(float))
    estimates = [float(hits[n - 1] / n) for n in NS]
    write_table(
        pd.DataFrame(
            {
                "replications": NS,
                "mc_estimate": estimates,
                "exact": exact,
                "abs_error": [abs(e - exact) for e in estimates],
            }
        ),
        out_dir,
        "e06_convergence",
    )
    parity = compare(ht, paths, thresholds=(60, 90, 110))
    write_table(parity, out_dir, "e06_parity")

    fig, ax = new_figure()
    ax.plot(NS, estimates, marker="o", label="Monte Carlo estimate", color=PALETTE[0])
    ax.axhline(exact, label="exact (forward recursion)", color=PALETTE[1])
    ax.set_xscale("log")
    ax.set_xlabel("replications")
    ax.set_ylabel(f"P(T ≥ {THRESHOLD})")
    ax.legend()
    save(fig, figures, "e06_fig_convergence")
    return {
        "exact_survival_90": exact,
        "mc_200k": estimates[-1],
        "max_abs_z": float(parity["z"].abs().max()),
    }
