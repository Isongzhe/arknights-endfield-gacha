"""Kernel invariants on every real model, so a new mechanic cannot break them unnoticed."""

import numpy as np
import pytest

from gacha.kernel.backward import state_values, success_within
from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.endfield import BannerState, SingleBannerModel
from gacha.models.paper import Featured5050Model, SingleCounterModel
from gacha.models.plan import Plan
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules, rerun_rules
from gacha.rules.paper import PaperSchedule

FULL = EndfieldCharacterRules()
MODELS = {
    "paper_single": lambda: SingleCounterModel(PaperSchedule().probs),
    "paper_featured": lambda: Featured5050Model(PaperSchedule().probs),
    "endfield_first_up": lambda: SingleBannerModel(BannerSpec(FULL, 1, 120)),
    "endfield_two_copies_capped": lambda: SingleBannerModel(BannerSpec(FULL, 2, 120)),
    "rerun_resumed": lambda: SingleBannerModel(
        BannerSpec(rerun_rules(), 1, 120), start=BannerState(30, 30, 0, 0)
    ),
    "plan_want_skip_want": lambda: Plan(
        [BannerSpec(FULL, 1, 120), BannerSpec(FULL, 0, 0), BannerSpec(FULL, 1, 60)]
    ),
}


@pytest.mark.parametrize("name", MODELS)
def test_mass_moments_and_budget_dp_agree(name):
    chain = EnumeratedChain.from_model(MODELS[name]())
    ht = hitting_time(chain)
    assert ht.f_succ.sum() + ht.f_fail.sum() + ht.residual == pytest.approx(1.0, abs=1e-12)
    assert ht.residual == pytest.approx(0.0, abs=1e-12)
    sv = state_values(chain)
    root = chain.roots[0]
    j = np.arange(ht.horizon + 1)
    mean = float((j * ht.pmf_stop).sum())
    assert sv.expectation[root] == pytest.approx(mean, abs=1e-9)
    assert sv.variance[root] == pytest.approx(
        float(((j - mean) ** 2 * ht.pmf_stop).sum()), abs=1e-7
    )
    assert sv.p_success[root] == pytest.approx(ht.p_success, abs=1e-12)
    within = success_within(chain, ht.horizon)[:, root]
    assert within.tolist() == pytest.approx(np.cumsum(ht.f_succ).tolist(), abs=1e-12)
