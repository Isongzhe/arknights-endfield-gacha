"""E1: reproduce the figures and tables of Hou, Zhu & Zhang (2026) as an engine validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gacha.kernel.backward import state_values
from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.paper import Featured5050Model, SingleCounterModel
from gacha.plots.style import apply_style, new_figure, save
from gacha.plots.waiting_time import plot_cdf, plot_pmf, plot_schedule, plot_survival
from gacha.risk import metrics as rk
from gacha.rules.paper import PaperSchedule

from ._io import ensure_dirs, write_table

ALPHAS = (0.5, 0.75, 0.9, 0.95, 0.99)
BUDGETS = (600, 650, 700, 719, 741, 775, 800)


def run(out_dir: Path) -> dict[str, float]:
    apply_style()
    _, figures = ensure_dirs(out_dir)
    sched = PaperSchedule()
    hard = sched.hard_only()
    soft_chain = EnumeratedChain.from_model(SingleCounterModel(sched.probs))
    soft = hitting_time(soft_chain)
    hard_ht = hitting_time(EnumeratedChain.from_model(SingleCounterModel(hard.probs)))
    t10 = rk.from_pmf(rk.convolve(soft.pmf_stop, 10))
    feat = hitting_time(EnumeratedChain.from_model(Featured5050Model(sched.probs)))
    sv = state_values(soft_chain)

    write_table(
        pd.DataFrame(
            {
                "model": ["uncapped geometric", "hard pity only (90)", "soft-pity schedule (1)"],
                "expected_draws": [1.0 / sched.base, rk.mean(hard_ht), rk.mean(soft)],
            }
        ),
        out_dir,
        "e01_baselines",
    )
    ts = [0, 70, 72, 73, 80, 89]
    write_table(
        pd.DataFrame(
            {
                "pity_t": ts,
                "expected_remaining": [sv.expectation[soft_chain.index[(t,)]] for t in ts],
            }
        ),
        out_dir,
        "e01_expected_remaining",
    )
    write_table(
        pd.DataFrame(
            {
                "alpha": ALPHAS,
                "VaR": [rk.quantile(t10, a) for a in ALPHAS],
                "CVaR": [rk.cvar(t10, a) for a in ALPHAS],
                "tail_prob_at_VaR": [rk.survival(t10, rk.quantile(t10, a)) for a in ALPHAS],
            }
        ),
        out_dir,
        "e01_t10_var_cvar",
    )
    write_table(
        pd.DataFrame(
            {
                "budget": BUDGETS,
                "completion": [rk.completion(t10, b) for b in BUDGETS],
                "expected_excess": [rk.expected_excess(t10, b) for b in BUDGETS],
            }
        ),
        out_dir,
        "e01_t10_budget",
    )

    fig, ax = new_figure()
    plot_schedule(ax, hard.probs, label="hard pity only")
    plot_schedule(ax, sched.probs, label="soft-pity schedule")
    ax.set_title("Paper schedule: state-dependent success probability")
    ax.legend()
    save(fig, figures, "e01_fig_schedule")

    fig, ax = new_figure()
    plot_pmf(ax, soft.pmf_stop)
    ax.set_title("Single-stage waiting-time PMF (paper Fig. 6)")
    save(fig, figures, "e01_fig_single_pmf")

    fig, ax = new_figure()
    plot_cdf(ax, soft.pmf_stop, label="CDF")
    plot_survival(ax, soft.pmf_stop, label="survival P(T ≥ b)")
    ax.set_ylabel("probability")
    ax.legend()
    save(fig, figures, "e01_fig_single_cdf")

    fig, ax = new_figure()
    plot_pmf(ax, t10.pmf_stop)
    ax.set_xlim(400, 900)
    ax.set_title("Ten independent rare items (paper Fig. 8)")
    save(fig, figures, "e01_fig_t10_pmf")

    levels = np.linspace(0.5, 0.999, 200)
    fig, ax = new_figure()
    ax.plot(levels, [rk.quantile(t10, a) for a in levels])
    ax.set_xlabel("quantile level")
    ax.set_ylabel("paid pulls")
    ax.set_title("Exact quantile curve for T10 (paper Fig. 9)")
    save(fig, figures, "e01_fig_t10_quantiles")

    fig, ax = new_figure()
    plot_survival(ax, t10.pmf_stop)
    ax.set_xlim(500, 850)
    ax.set_title("Right-tail P(T10 ≥ b) (paper Fig. 10)")
    save(fig, figures, "e01_fig_t10_tail")

    levels = np.linspace(0.7, 0.995, 60)
    fig, ax = new_figure()
    ax.plot(levels, [rk.quantile(t10, a) for a in levels], label="VaR")
    ax.plot(levels, [rk.cvar(t10, a) for a in levels], label="CVaR (tail conditional mean)")
    ax.set_xlabel("risk level α")
    ax.set_ylabel("paid pulls")
    ax.legend()
    save(fig, figures, "e01_fig_t10_var_cvar")

    fig, ax = new_figure()
    plot_pmf(ax, soft.pmf_stop, label="any rare item")
    plot_pmf(ax, feat.pmf_stop, label="featured target, no guarantee")
    ax.legend()
    ax.set_title("Effect of a 50/50 guarantee state (paper Fig. 19)")
    save(fig, figures, "e01_fig_featured_pmf")

    return {
        "E0": rk.mean(soft),
        "V0": rk.var(soft),
        "hard_only_E0": rk.mean(hard_ht),
        "T10_mean": rk.mean(t10),
        "T10_q90": rk.quantile(t10, 0.9),
        "featured_mean": rk.mean(feat),
    }
