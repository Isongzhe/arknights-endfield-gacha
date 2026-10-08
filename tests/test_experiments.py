from pathlib import Path

import pytest

from gacharisk.experiments import EXPERIMENTS, run_experiment


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


def test_e03_personal_state(tmp_path: Path):
    head = run_experiment("e03_personal_state", tmp_path)
    assert head["E_t79_n0"] < head["E_fresh"]
    assert head["E_t0_n100"] < head["E_fresh"]
    assert head["monotone_in_t0"] == 1.0
    assert "e03_fig_expected_remaining.png" in _files(tmp_path, "figures", "e03")
    assert "e03_value_of_pity.csv" in _files(tmp_path, "tables", "e03")


def test_e04_carry_over(tmp_path: Path):
    head = run_experiment("e04_carry_over", tmp_path)
    assert head["gap_K2"] > 1e-3 and head["gap_K3"] > 1e-3
    assert head["exact_mean_K2"] < head["iid_mean_K2"]  # entering pity and dossier help
    assert "e04_carry_over.md" in _files(tmp_path, "tables", "e04")
    assert "e04_fig_cdf.png" in _files(tmp_path, "figures", "e04")


def test_e05_tail_risk(tmp_path: Path):
    head = run_experiment("e05_tail_risk", tmp_path)
    assert head["q90_K1"] < head["q90_K3"] < head["q90_K5"]
    assert 0.0 < head["completion_60_per_banner_K5"] < 1.0
    assert "e05_tail_risk.csv" in _files(tmp_path, "tables", "e05")
    assert "e05_fig_completion.png" in _files(tmp_path, "figures", "e05")


def test_e06_mc_convergence(tmp_path: Path):
    head = run_experiment("e06_mc_convergence", tmp_path)
    assert abs(head["mc_200k"] - head["exact_survival_90"]) < 0.005
    assert head["max_abs_z"] < 3.5
    assert "e06_convergence.md" in _files(tmp_path, "tables", "e06")
    assert "e06_fig_convergence.png" in _files(tmp_path, "figures", "e06")


def test_e07_sensitivity(tmp_path: Path):
    head = run_experiment("e07_sensitivity", tmp_path)
    assert head["mean_up_share_0.7"] < head["mean_up_share_0.5"]
    assert head["mean_guarantee_100"] < head["mean_guarantee_140"]
    assert "e07_sensitivity.md" in _files(tmp_path, "tables", "e07")
    assert "e07_fig_sensitivity.png" in _files(tmp_path, "figures", "e07")


def test_registry_complete():
    assert EXPERIMENTS == [
        "e01_reproduce_paper",
        "e02_endfield_single",
        "e03_personal_state",
        "e04_carry_over",
        "e05_tail_risk",
        "e06_mc_convergence",
        "e07_sensitivity",
    ]
