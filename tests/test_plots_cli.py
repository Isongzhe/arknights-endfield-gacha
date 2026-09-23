import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gacha.cli import main
from gacha.experiments import EXPERIMENTS, run_experiment
from gacha.experiments._io import ensure_dirs, update_results_md, write_table
from gacha.plots.heatmaps import heatmap
from gacha.plots.style import PALETTE, apply_style, new_figure, save
from gacha.plots.waiting_time import plot_cdf, plot_pmf, plot_schedule, plot_survival


def test_palette_and_figure_roundtrip(tmp_path: Path):
    assert len(PALETTE) == 8 and PALETTE[0] == "#2a78d6"
    apply_style()
    fig, ax = new_figure()
    pmf = np.array([0.0, 0.5, 0.3, 0.2])
    plot_pmf(ax, pmf, label="a")
    plot_cdf(ax, pmf, label="b")
    plot_survival(ax, pmf, label="c")
    plot_schedule(ax, [0.1, 0.5, 1.0], label="d")
    ax.legend()
    paths = save(fig, tmp_path, "fig")
    assert [p.name for p in paths] == ["fig.png", "fig.pdf"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_heatmap_draws(tmp_path: Path):
    apply_style()
    fig, ax = new_figure()
    z = np.arange(12, dtype=float).reshape(3, 4)
    im = heatmap(
        ax,
        z,
        x=np.array([0, 5, 10, 15]),
        y=np.array([1, 2, 3]),
        xlabel="x",
        ylabel="y",
        cbar_label="z",
    )
    assert im.get_array().shape == (3, 4)
    save(fig, tmp_path, "hm")


def test_write_table_and_results_md(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2], "b": [0.123456, 2.5]})
    write_table(df, tmp_path, "t")
    tables, figures = ensure_dirs(tmp_path)
    assert (tables / "t.csv").exists() and (tables / "t.md").exists()
    assert "| a | b |" in (tables / "t.md").read_text()
    md = tmp_path / "results.md"
    update_results_md(md, "e00", ["- x: 1"])
    update_results_md(md, "e01", ["- y: 2"])
    update_results_md(md, "e00", ["- x: 3"])
    text = md.read_text()
    assert "## e00\n\n- x: 3" in text and "- x: 1" not in text and "## e01\n\n- y: 2" in text


def test_registry_is_a_list_of_strings():
    assert isinstance(EXPERIMENTS, list)
    with pytest.raises(KeyError):
        run_experiment("does_not_exist", Path("."))


def test_cli_evaluate_prints_metrics(capsys):
    rc = main(
        ["evaluate", "--pity", "70", "--banner-pulls", "50", "--budget", "30", "--realized", "60"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "P(success within 30 paid)" in out
    assert "P(T >= 60) from a fresh banner" in out


def test_cli_evaluate_already_owned(capsys):
    rc = main(["evaluate", "--copies", "1", "--up-obtained", "1", "--budget", "999"])
    out = capsys.readouterr().out
    assert rc == 0
    assert re.search(r"P\(success\)\s+1\.0000", out)
    assert re.search(r"P\(success within 999 paid\)\s+1\.0000", out)


def test_cli_evaluate_plan_file(tmp_path: Path, capsys):
    plan = tmp_path / "plan.toml"
    plan.write_text(
        "dossier0 = false\n[start]\nt = 10\n[[banners]]\ntarget_copies = 1\ncap = 120\n"
        "[[banners]]\ntarget_copies = 0\ncap = 0\n[[banners]]\ntarget_copies = 1\ncap = 120\n"
    )
    rc = main(["evaluate", "--plan", str(plan), "--budget", "150"])
    assert rc == 0
    assert "P(success within 150 paid)" in capsys.readouterr().out


def test_cli_unknown_experiment(capsys):
    assert main(["experiment", "nope"]) == 2
    assert "unknown experiment" in capsys.readouterr().err
