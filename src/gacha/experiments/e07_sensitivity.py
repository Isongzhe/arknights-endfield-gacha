"""E7: sensitivity of the first-UP distribution to UP share, a soft-pity ramp and the guarantee."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.endfield import SingleBannerModel
from gacha.plots.style import apply_style, new_figure, save
from gacha.risk import metrics as rk
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules

from ._io import ensure_dirs, write_table

GROUPS = ("up_share", "soft_pity_step", "guarantee_pull")


def run(out_dir: Path) -> dict[str, float]:
    apply_style()
    _, figures = ensure_dirs(out_dir)
    base = EndfieldCharacterRules()
    rows = []

    def evaluate(group: str, label: str, rules: EndfieldCharacterRules, cap: int) -> None:
        model = SingleBannerModel(BannerSpec(rules, 1, cap))
        ht = hitting_time(EnumeratedChain.from_model(model))
        rows.append(
            {
                "group": group,
                "setting": label,
                "mean": rk.mean(ht),
                "q90": rk.quantile(ht, 0.9),
                "q99": rk.quantile(ht, 0.99),
                "p_success": ht.p_success,
            }
        )

    for q in (0.5, 0.6, 0.7):
        evaluate("up_share", f"up_share={q}", replace(base, up_share=q), 120)
    for s in (0.05, 0.0):  # 0.0 = no soft-pity ramp, only the 80th-pull hard pity
        evaluate("soft_pity_step", f"soft_pity_step={s}", replace(base, soft_pity_step=s), 120)
    for g in (100, 120, 140, None):
        evaluate("guarantee_pull", f"guarantee={g}", replace(base, guarantee_pull=g), g or 240)
    df = pd.DataFrame(rows)
    write_table(df, out_dir, "e07_sensitivity")

    fig, axes = new_figure(11, 3.4, ncols=3)
    for ax, group in zip(axes, GROUPS, strict=True):
        sub = df[df["group"] == group]
        x = list(range(len(sub)))
        ax.plot(x, sub["mean"], marker="o", label="mean")
        ax.plot(x, sub["q90"], marker="s", label="90% quantile")
        ax.set_xticks(x, sub["setting"], rotation=20)
        ax.set_title(group)
    axes[0].set_ylabel("paid pulls")
    axes[0].legend()
    save(fig, figures, "e07_fig_sensitivity")

    by = {r["setting"]: r for r in rows}
    return {
        "mean_up_share_0.5": by["up_share=0.5"]["mean"],
        "mean_up_share_0.7": by["up_share=0.7"]["mean"],
        "mean_no_ramp": by["soft_pity_step=0.0"]["mean"],
        "mean_guarantee_100": by["guarantee=100"]["mean"],
        "mean_guarantee_140": by["guarantee=140"]["mean"],
        "mean_guarantee_none": by["guarantee=None"]["mean"],
    }
