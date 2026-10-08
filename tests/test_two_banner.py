from pathlib import Path

import numpy as np
import pytest

from gacharisk.analysis.two_banner import cap_table, first_up_pmf, recommend
from gacharisk.cli import main
from gacharisk.models.endfield import BannerState, SingleBannerModel
from gacharisk.rules.endfield import BannerSpec, EndfieldCharacterRules, rerun_rules
from gacharisk.scenario import load_scenario

EXAMPLE = Path(__file__).parent.parent / "examples" / "rerun_then_limited.toml"


@pytest.fixture(scope="module")
def pmfs():
    a = first_up_pmf(
        SingleBannerModel(BannerSpec(rerun_rules(), 1, 120), start=BannerState(30, 30, 0, 0))
    )
    rules = EndfieldCharacterRules(free_start_pulls=10)
    b = first_up_pmf(SingleBannerModel(BannerSpec(rules, 1, 120), start=BannerState(48, 0, 0, 0)))
    return a, b


def test_first_up_pmf_is_a_proper_distribution(pmfs):
    a, b = pmfs
    assert a.sum() == pytest.approx(1.0) and b.sum() == pytest.approx(1.0)
    assert len(a) - 1 == 90 and len(b) - 1 == 110  # pulls to each banner's guarantee


def test_cap_table_reference_values(pmfs):
    rows = cap_table(*pmfs, stock=89)
    assert len(rows) == 90
    assert rows[0].p_first == 0.0 and rows[0].p_both == 0.0
    assert rows[0].p_second == pytest.approx(0.782, abs=5e-4)
    r = rows[50]
    assert (r.p_first, r.p_second, r.p_both) == pytest.approx((0.5481, 0.5901, 0.3318), abs=5e-4)


def test_all_in_equals_convolution(pmfs):
    a, b = pmfs
    rows = cap_table(a, b, stock=150)
    total = np.cumsum(np.convolve(a, b))
    assert rows[-1].p_both == pytest.approx(total[150], abs=1e-12)
    assert all(r.p_both <= min(r.p_first, r.p_second) + 1e-12 for r in rows)


def test_recommend_prefers_the_smallest_cap_within_tolerance(pmfs):
    rows = cap_table(*pmfs, stock=89)
    best = max(r.p_both for r in rows)
    pick = recommend(rows, tolerance=0.01)
    assert pick.p_both >= best - 0.01
    assert all(r.p_both < best - 0.01 for r in rows[: pick.cap])
    with pytest.raises(ValueError):
        cap_table(*pmfs, stock=-1)


def test_load_scenario_from_example():
    sc = load_scenario(EXAMPLE)
    assert sc.stock_now == 78 and sc.stock_total == 89 and sc.leftover_jade == 175
    assert sc.first.start == BannerState(30, 30, 0, 0)
    assert sc.second.start == BannerState(48, 0, 0, 0)
    assert sc.second.rules.free_start_pulls == 10
    assert sc.first.rules.vacuum_points == (30, 60, 90)


def test_cli_decide_prints_table_and_cost(capsys):
    assert main(["decide", str(EXAMPLE)]) == 0
    out = capsys.readouterr().out
    assert "89" in out and "54.8%" in out and "59.0%" in out and "33.2%" in out
    assert "NT$6,120" in out  # worst-case top-up to guarantee both


def test_cli_reports_bad_input_without_traceback(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text('[stock]\njade = 0\n[first]\nkind = "weapon"\npulls_to_six_star = 50\n')
    assert main(["decide", str(bad)]) == 2
    assert "error:" in capsys.readouterr().err
    assert main(["decide", str(tmp_path / "missing.toml")]) == 2
    assert main(["evaluate", "--pity", "80"]) == 2
    assert "error:" in capsys.readouterr().err
