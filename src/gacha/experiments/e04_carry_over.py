"""E4: exact multi-banner distribution (pity and dossier carried) versus the paper's
i.i.d. convolution (its Eq. 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.endfield import SingleBannerModel
from gacha.models.plan import Plan
from gacha.plots.style import PALETTE, apply_style, new_figure, save
from gacha.risk import metrics as rk
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules

from ._io import ensure_dirs, write_table

KS = (2, 3)


def _padded_cdfs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = max(len(a), len(b))
    return np.cumsum(np.pad(a, (0, n - len(a)))), np.cumsum(np.pad(b, (0, n - len(b))))


def run(out_dir: Path) -> dict[str, float]:
    apply_style()
    _, figures = ensure_dirs(out_dir)
    spec = BannerSpec(EndfieldCharacterRules(), 1, 120)
    single = hitting_time(EnumeratedChain.from_model(SingleBannerModel(spec)))
    rows, curves = [], {}
    for k in KS:
        exact = hitting_time(EnumeratedChain.from_model(Plan([spec] * k)))
        iid = rk.from_pmf(rk.convolve(single.pmf_stop, k))
        ce, cc = _padded_cdfs(exact.pmf_stop, iid.pmf_stop)
        curves[k] = (ce, cc)
        rows.append(
            {
                "banners": k,
                "exact_mean": rk.mean(exact),
                "iid_mean": rk.mean(iid),
                "exact_q50": rk.quantile(exact, 0.5),
                "iid_q50": rk.quantile(iid, 0.5),
                "exact_q90": rk.quantile(exact, 0.9),
                "iid_q90": rk.quantile(iid, 0.9),
                "exact_q95": rk.quantile(exact, 0.95),
                "iid_q95": rk.quantile(iid, 0.95),
                "max_cdf_gap": float(np.abs(ce - cc).max()),
                "exact_p_within_100_per_banner": rk.completion(exact, 100 * k),
                "iid_p_within_100_per_banner": rk.completion(iid, 100 * k),
            }
        )
    write_table(pd.DataFrame(rows), out_dir, "e04_carry_over")

    fig, axes = new_figure(10, 3.6, ncols=len(KS))
    for ax, k in zip(axes, KS, strict=True):
        ce, cc = curves[k]
        ax.plot(ce, label="exact (pity + dossier carried)", color=PALETTE[0])
        ax.plot(cc, label="i.i.d. convolution (paper Eq. 2)", color=PALETTE[1], linestyle="--")
        ax.set_title(f"{k} consecutive wanted banners")
        ax.set_xlabel("paid pulls b")
        ax.set_ylabel("P(all UPs, T ≤ b)")
    axes[0].legend()
    save(fig, figures, "e04_fig_cdf")

    fig, ax = new_figure()
    for k in KS:
        ce, cc = curves[k]
        ax.plot(ce - cc, label=f"{k} banners")
    ax.set_xlabel("paid pulls b")
    ax.set_ylabel("exact CDF − i.i.d. CDF")
    ax.legend()
    save(fig, figures, "e04_fig_gap")

    by_k = {int(r["banners"]): r for r in rows}
    return {
        "gap_K2": by_k[2]["max_cdf_gap"],
        "gap_K3": by_k[3]["max_cdf_gap"],
        "exact_mean_K2": by_k[2]["exact_mean"],
        "iid_mean_K2": by_k[2]["iid_mean"],
        "exact_q90_K3": by_k[3]["exact_q90"],
        "iid_q90_K3": by_k[3]["iid_q90"],
    }
