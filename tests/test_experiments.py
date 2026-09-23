from pathlib import Path

import pytest

from gacha.experiments import EXPERIMENTS, run_experiment


def _files(out: Path, kind: str, prefix: str) -> list[str]:
    return sorted(p.name for p in (out / kind).glob(f"{prefix}*"))


def test_e01_reproduce_paper(tmp_path: Path):
    head = run_experiment("e01_reproduce_paper", tmp_path, results_md=tmp_path / "results.md")
    assert head["E0"] == pytest.approx(62.34, abs=0.01)
    assert head["T10_q90"] == 719
    assert head["featured_mean"] == pytest.approx(93.51, abs=0.01)
    assert "e01_baselines.csv" in _files(tmp_path, "tables", "e01")
    assert "e01_fig_t10_pmf.png" in _files(tmp_path, "figures", "e01")
    assert "## e01_reproduce_paper" in (tmp_path / "results.md").read_text()


def test_e02_endfield_single(tmp_path: Path):
    head = run_experiment("e02_endfield_single", tmp_path)
    assert head["full_support_max"] == 115
    assert 40 < head["full_mean"] < 90
    assert head["two_copies_p_success"] == pytest.approx(1.0, abs=1e-9)
    assert "e02_variants.md" in _files(tmp_path, "tables", "e02")
    assert "e02_fig_pmf.pdf" in _files(tmp_path, "figures", "e02")


def test_registry_lists_e01_and_e02():
    assert "e01_reproduce_paper" in EXPERIMENTS and "e02_endfield_single" in EXPERIMENTS
