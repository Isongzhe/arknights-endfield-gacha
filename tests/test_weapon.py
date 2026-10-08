import pytest

from gacharisk.analysis.rebate import arsenal_per_pull
from gacharisk.cli import main
from gacharisk.kernel.chain import EnumeratedChain
from gacharisk.kernel.forward import hitting_time
from gacharisk.models.weapon import WeaponBannerModel, WeaponState
from gacharisk.risk import metrics as rk
from gacharisk.rules.endfield import EndfieldCharacterRules
from gacharisk.rules.weapon import WeaponBannerRules

RULES = WeaponBannerRules()


def ht_of(**kw):
    return hitting_time(EnumeratedChain.from_model(WeaponBannerModel(RULES, **kw)))


def test_rules_defaults_and_validation():
    assert RULES.issue_cost == 1980 and RULES.pulls_per_issue == 10
    assert RULES.max_issues == 8
    with pytest.raises(ValueError):
        WeaponBannerRules(up_guarantee=85)  # must be a whole number of issues
    with pytest.raises(ValueError):
        WeaponBannerRules(six_rate=1.5)


def test_first_issue_and_guarantee():
    ht = ht_of()
    assert ht.horizon == 8 and ht.p_success == pytest.approx(1.0)
    assert ht.f_succ[0] == 0.0
    assert ht.f_succ[1] == pytest.approx(1 - 0.99**10)  # ten pulls at 1% each
    cdf = ht.f_succ.cumsum()
    # pulls 1-39 at 1%, pull 40 is a forced 6* for those with none so far
    assert cdf[3] == pytest.approx(1 - 0.99**30)
    assert [round(float(c), 3) for c in cdf[1:]] == [
        0.096,
        0.182,
        0.26,
        0.38,
        0.453,
        0.517,
        0.574,
        1.0,
    ]
    assert rk.mean(ht) == pytest.approx(5.54, abs=0.01)


def test_forty_pull_pity_forces_a_six_star():
    m = WeaponBannerModel(RULES)
    [(p_up, nxt_up), (p_off, nxt_off)] = sorted(
        ((t.prob, t.next) for t in m.step(WeaponState(39, 39))), key=lambda x: x[0]
    )
    assert p_up == pytest.approx(0.25) and nxt_up is None  # absorbed: UP weapon
    assert p_off == pytest.approx(0.75) and nxt_off == WeaponState(40, 0)


def test_resuming_and_caps():
    resumed = ht_of(start=WeaponState(40, 0))
    assert resumed.horizon == 4 and resumed.p_success == pytest.approx(1.0)
    capped = ht_of(cap_issues=3)
    assert capped.p_success == pytest.approx(1 - 0.99**30)
    assert capped.p_fail == pytest.approx(0.99**30)
    with pytest.raises(ValueError):
        WeaponBannerModel(RULES, start=WeaponState(15, 0))  # must start on an issue boundary
    with pytest.raises(ValueError):
        WeaponBannerModel(RULES, start=WeaponState(80, 0))


def test_arsenal_quota_from_character_pulls():
    per = arsenal_per_pull(EndfieldCharacterRules())
    assert per == pytest.approx(80.9, abs=0.2)  # 4* 20, 5* 200, 6* 2000


def test_cli_weapon(capsys):
    assert main(["weapon", "--quota", "43940"]) == 0
    out = capsys.readouterr().out
    assert "22" in out and "100.0%" in out
    assert main(["weapon", "--quota", "5000"]) == 0
    out = capsys.readouterr().out
    assert "18.2%" in out  # two issues affordable
    assert main(["weapon", "--quota", "100", "--issues-done", "9"]) == 2
