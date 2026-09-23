import numpy as np
import pytest

from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.kernel.types import FAIL, SUCCESS
from gacha.models.endfield import BannerState, SingleBannerModel
from gacha.models.plan import Plan, PlanState
from gacha.risk import metrics as rk
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules

FULL = EndfieldCharacterRules()
WANTED = BannerSpec(FULL, 1, 120)
SKIP = BannerSpec(FULL, 0, 0)


def ht_of(model):
    return hitting_time(EnumeratedChain.from_model(model))


def test_plan_rejects_empty():
    with pytest.raises(ValueError):
        Plan([])


def test_one_banner_plan_equals_single_banner():
    single = ht_of(SingleBannerModel(WANTED))
    plan = ht_of(Plan([WANTED]))
    assert plan.f_succ.tolist() == pytest.approx(single.f_succ.tolist(), abs=1e-12)
    assert plan.f_fail.tolist() == pytest.approx(single.f_fail.tolist(), abs=1e-12)


def test_boundary_transitions():
    plan = Plan([WANTED, WANTED])
    # banner 0 finished with n >= 60: dossier granted to banner 1
    [tr] = plan.step(PlanState(0, 3, 61, 1, 1, 0))
    assert tr.next == PlanState(1, 3, 0, 0, 0, 1) and tr.cost == 0 and tr.prob == 1.0
    [tr] = plan.step(PlanState(0, 3, 59, 1, 1, 0))
    assert tr.next == PlanState(1, 3, 0, 0, 0, 0)
    # wanted banner exhausted -> fail; last banner success -> success
    [tr] = plan.step(PlanState(0, 3, 125, 0, 0, 0))
    assert tr.absorb == FAIL
    [tr] = plan.step(PlanState(1, 3, 20, 1, 1, 1))
    assert tr.absorb == SUCCESS


def test_skipped_banner_only_uses_free_pulls():
    ht = ht_of(Plan([SKIP, WANTED]))
    # success with 0 paid pulls needs a UP among banner 2's 5 free pulls; banner 1's free
    # pulls only move the pity counter inside the flat region, so the probability is flat.
    assert ht.f_succ[0] == pytest.approx(1 - (1 - 0.004) ** 5, abs=1e-12)
    assert ht.horizon == 115


def test_plan_last_skipped_banner_succeeds():
    ht = ht_of(Plan([WANTED, SKIP]))
    assert ht.p_success == pytest.approx(1.0, abs=1e-12)
    assert ht.horizon == 115


def test_dossier_gives_next_banner_fifteen_free_pulls():
    # start banner 0 already at n=60 (dossier earned) with the UP in hand: banner 1 gets f=15
    plan = Plan([WANTED, WANTED], start=BannerState(0, 60, 1, 1))
    ht = ht_of(plan)
    fresh15 = ht_of(SingleBannerModel(WANTED, dossier=True))
    assert ht.horizon == fresh15.horizon == 105
    assert ht.f_succ.tolist() == pytest.approx(fresh15.f_succ.tolist(), abs=1e-12)


def test_cross_banner_coupling_differs_from_iid_convolution():
    single = ht_of(SingleBannerModel(WANTED))
    exact = ht_of(Plan([WANTED, WANTED]))
    conv = rk.from_pmf(rk.convolve(single.pmf_stop, 2))
    n = max(exact.horizon, conv.horizon) + 1
    ce = np.cumsum(np.pad(exact.pmf_stop, (0, n - exact.horizon - 1)))
    cc = np.cumsum(np.pad(conv.pmf_stop, (0, n - conv.horizon - 1)))
    assert np.abs(ce - cc).max() > 1e-3
    assert exact.p_success == pytest.approx(1.0, abs=1e-12)


def test_validates_start_against_first_banner():
    with pytest.raises(ValueError):
        Plan([WANTED], start=BannerState(0, 0, 3, 1))
