import math

import pytest

from gacharisk.rules.endfield import BannerSpec, EndfieldCharacterRules
from gacharisk.rules.paper import PaperSchedule
from gacharisk.rules.schedule import ramp_schedule


def test_ramp_schedule_paper_values():
    p = ramp_schedule(0.006, 73, 0.0585, 90)
    assert len(p) == 90
    assert p[72] == pytest.approx(0.006)
    assert p[73] == pytest.approx(0.0645)
    assert p[88] == pytest.approx(0.006 + 16 * 0.0585)
    assert p[89] == 1.0


def test_ramp_schedule_no_ramp_when_step_zero():
    p = ramp_schedule(0.006, 73, 0.0, 90)
    assert all(x == pytest.approx(0.006) for x in p[:89])
    assert p[89] == 1.0


def test_ramp_schedule_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ramp_schedule(0.0, 1, 0.0, 10)
    with pytest.raises(ValueError):
        ramp_schedule(0.5, 1, 0.0, 0)
    with pytest.raises(ValueError):
        ramp_schedule(0.5, 1, -0.1, 10)


def test_paper_schedule_defaults_and_hard_only():
    s = PaperSchedule()
    assert s.p(0) == pytest.approx(0.006)
    assert s.p(89) == 1.0
    h = s.hard_only()
    assert h.p(88) == pytest.approx(0.006)
    assert h.p(89) == 1.0


def test_endfield_schedule_key_values():
    r = EndfieldCharacterRules()
    assert len(r.probs) == 80
    # R2: +5% per pull from the 66th pull (t = 65), forced on the 80th.
    assert r.soft_pity_step == 0.05
    assert r.p(64) == pytest.approx(0.008)
    assert r.p(65) == pytest.approx(0.058)
    assert r.p(78) == pytest.approx(0.708)
    assert r.p(79) == 1.0
    flat = EndfieldCharacterRules(soft_pity_step=0.0)  # the no-ramp variant stays expressible
    assert flat.p(78) == pytest.approx(0.008)


def test_endfield_free_pulls():
    r = EndfieldCharacterRules()
    assert r.free_pulls(False) == 5
    assert r.free_pulls(True) == 15
    assert EndfieldCharacterRules(free_start_pulls=0).free_pulls(True) == 10


def test_vacuum_copies_pmf_is_binomial():
    r = EndfieldCharacterRules()
    pmf = r.vacuum_copies_pmf()
    assert len(pmf) == 11
    assert sum(pmf) == pytest.approx(1.0)
    assert pmf[0] == pytest.approx((1 - 0.008 * 0.5) ** 10)
    assert pmf[1] == pytest.approx(10 * 0.004 * 0.996**9)
    assert EndfieldCharacterRules(vacuum_at=None).vacuum_copies_pmf() == (1.0,)


def test_endfield_rules_validation():
    with pytest.raises(ValueError):
        EndfieldCharacterRules(up_share=1.5)
    with pytest.raises(ValueError):
        EndfieldCharacterRules(guarantee_pull=0)
    with pytest.raises(ValueError):
        EndfieldCharacterRules(free_start_pulls=-1)


def test_banner_spec_validation():
    r = EndfieldCharacterRules()
    BannerSpec(r, target_copies=1, cap=120)
    BannerSpec(r, target_copies=0, cap=0)
    with pytest.raises(ValueError):
        BannerSpec(r, target_copies=0, cap=10)
    with pytest.raises(ValueError):
        BannerSpec(r, target_copies=-1, cap=0)
    with pytest.raises(ValueError):
        BannerSpec(r, target_copies=1, cap=-5)
    assert math.isfinite(r.p(0))


def test_multiple_vacuum_points_and_rerun_rules():
    from gacharisk.rules.endfield import rerun_rules

    assert EndfieldCharacterRules().vacuum_points == (30,)
    assert EndfieldCharacterRules(vacuum_at=None).vacuum_points == ()
    r = rerun_rules()
    assert r.vacuum_points == (30, 60, 90)  # RR3
    assert r.dossier_at is None and r.free_start_pulls == 0  # RR7
    assert r.guarantee_pull == 120 and r.potential_every == 240
    assert len(r.vacuum_copies_pmf()) == 11
    with pytest.raises(ValueError):
        EndfieldCharacterRules(vacuum_at=(30, 0))
