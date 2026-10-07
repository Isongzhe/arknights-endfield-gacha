from dataclasses import replace

import pytest

from gacha.kernel.backward import state_values
from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.endfield import BannerState, SingleBannerModel, pull, status, validate_start
from gacha.risk import metrics as rk
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules

FULL = EndfieldCharacterRules()
# Only the 80-pull pity and the 50/50 (R2-R4); everything banner-local switched off.
BARE = EndfieldCharacterRules(
    guarantee_pull=None, vacuum_at=None, dossier_at=None, potential_every=None, free_start_pulls=0
)


def ht_of(spec, **kw):
    return hitting_time(EnumeratedChain.from_model(SingleBannerModel(spec, **kw)))


def test_pull_branches_and_costs():
    f = FULL.free_pulls(False)  # 5
    trs = pull(FULL, BannerState(0, 0, 0, 0), f, 1)
    assert all(cost == 0 for _, _, cost in trs)  # first pull is free
    probs = {nxt: p for p, nxt, _ in trs}
    assert probs[BannerState(0, 1, 1, 1)] == pytest.approx(0.004)  # UP
    assert probs[BannerState(0, 1, 0, 0)] == pytest.approx(0.004)  # off-rate 6*
    assert probs[BannerState(1, 1, 0, 0)] == pytest.approx(0.992)
    trs = pull(FULL, BannerState(10, 5, 0, 0), f, 1)
    assert all(cost == 1 for _, _, cost in trs)  # sixth counted pull is paid


def test_pull_guarantee_forces_up_once():
    trs = pull(BARE, BannerState(3, 119, 0, 0), 0, 1)
    assert len(trs) == 3  # guarantee disabled in BARE
    rules = replace(BARE, guarantee_pull=120)
    trs = pull(rules, BannerState(3, 119, 0, 0), 0, 1)
    assert trs == [(1.0, BannerState(0, 120, 1, 1), 1)]
    trs = pull(rules, BannerState(3, 119, 0, 1), 0, 1)  # already obtained: no force
    assert sum(p for p, nxt, _ in trs if nxt.c == 1) == pytest.approx(0.004)


def test_pull_vacuum_and_potential_bonus():
    rules = replace(BARE, vacuum_at=30)
    trs = pull(rules, BannerState(2, 29, 0, 0), 0, 1)
    p_vac = 1 - (1 - 0.004) ** 10  # P(at least one UP among the 10 vacuum pulls)
    reached = sum(p for p, nxt, _ in trs if nxt.c == 1)
    assert reached == pytest.approx(0.004 + 0.996 * p_vac)
    rules = replace(BARE, potential_every=240)
    trs = pull(rules, BannerState(0, 239, 1, 1), 0, 2)
    assert all(nxt.c == 2 for _, nxt, _ in trs)


def test_status_rules():
    spec = BannerSpec(FULL, target_copies=1, cap=10)
    assert status(FULL, BannerState(0, 0, 0, 0), spec, 5) == "active"
    assert status(FULL, BannerState(0, 3, 1, 1), spec, 5) == "active"  # free pulls remain (P3)
    assert status(FULL, BannerState(0, 5, 1, 1), spec, 5) == "success"
    assert status(FULL, BannerState(0, 15, 0, 0), spec, 5) == "exhausted"  # paid 10 == cap
    skip = BannerSpec(FULL, target_copies=0, cap=0)
    assert status(FULL, BannerState(0, 4, 0, 0), skip, 5) == "active"
    assert status(FULL, BannerState(0, 5, 0, 0), skip, 5) == "exhausted"


def test_validate_start_rejects_bad_states():
    spec = BannerSpec(FULL, 1, 120)
    validate_start(spec, BannerState(79, 0, 0, 0), 5)
    with pytest.raises(ValueError):
        validate_start(spec, BannerState(80, 0, 0, 0), 5)
    with pytest.raises(ValueError):
        validate_start(spec, BannerState(0, 0, 2, 1), 5)  # more copies than target
    with pytest.raises(ValueError):
        validate_start(spec, BannerState(0, 200, 0, 0), 5)  # beyond paid cap
    with pytest.raises(ValueError):
        validate_start(spec, BannerState(0, 0, 0, 2), 5)
    with pytest.raises(ValueError):
        SingleBannerModel(spec, start=BannerState(-1, 0, 0, 0))


def test_validate_start_rejects_pulls_past_guarantee_without_up():
    # The 120th counted pull always sets u = 1, so n >= 120 with u == 0 cannot occur in the game.
    spec = BannerSpec(FULL, 1, 200)
    with pytest.raises(ValueError):
        validate_start(spec, BannerState(0, 120, 0, 0), 5)
    with pytest.raises(ValueError):
        validate_start(spec, BannerState(0, 121, 0, 0), 5)
    validate_start(spec, BannerState(0, 121, 1, 1), 5)  # possible: UP obtained at pull 120
    validate_start(BannerSpec(BARE, 1, 200), BannerState(0, 121, 0, 0), 0)  # no guarantee rule


def test_first_six_star_closed_form():
    # up_share=1: first UP == first 6*, bounded by hard pity. Paper Eq. (8) survival sum with
    # the rule schedule: E = sum_{j<80} prod_{r<j} (1 - p_r).
    rules = replace(BARE, up_share=1.0)
    ht = ht_of(BannerSpec(rules, 1, 80))
    assert ht.p_success == pytest.approx(1.0, abs=1e-12)
    survival = [1.0]
    for r in range(79):
        survival.append(survival[-1] * (1.0 - rules.p(r)))
    assert rk.mean(ht) == pytest.approx(sum(survival), abs=1e-9)
    assert ht.f_succ[80] == pytest.approx(survival[79], abs=1e-12)
    assert rk.mean(ht) < sum(0.992**j for j in range(80))  # the ramp beats a flat schedule
    flat = ht_of(BannerSpec(replace(rules, soft_pity_step=0.0), 1, 80))
    assert rk.mean(flat) == pytest.approx(sum(0.992**j for j in range(80)), abs=1e-9)


def test_fifty_fifty_closed_form():
    # hard_pity=1: every pull is a 6*, so T ~ Geometric(0.5) truncated at the cap
    rules = replace(BARE, hard_pity=1, soft_pity_start=0, soft_pity_step=0.0)
    cap = 6
    ht = ht_of(BannerSpec(rules, 1, cap))
    expected = [0.0] + [0.5**j for j in range(1, cap + 1)]
    assert ht.f_succ.tolist() == pytest.approx(expected, abs=1e-12)
    assert ht.p_fail == pytest.approx(0.5**cap)
    assert rk.mean(ht) == pytest.approx(sum(j * 0.5**j for j in range(1, cap + 1)) + cap * 0.5**cap)


def test_support_bound_from_guarantee():
    ht = ht_of(BannerSpec(replace(FULL, free_start_pulls=0), 1, 120))
    assert ht.horizon == 120 and ht.f_succ[120] > 0 and ht.p_success == pytest.approx(1.0)
    ht = ht_of(BannerSpec(FULL, 1, 120), dossier=True)  # f = 15
    assert ht.horizon == 105 and ht.f_succ[105] > 0 and ht.p_success == pytest.approx(1.0)


def test_entering_pity_79_gives_immediate_coin_flip():
    ht = ht_of(BannerSpec(BARE, 1, 120), start=BannerState(79, 0, 0, 0))
    assert ht.f_succ[1] == pytest.approx(0.5)


def test_guarantee_void_after_up():
    rules = replace(BARE, guarantee_pull=120)
    voided = ht_of(BannerSpec(rules, 2, 200), start=BannerState(0, 119, 1, 1))
    assert voided.f_succ[1] == pytest.approx(0.004)
    live = ht_of(BannerSpec(rules, 1, 200), start=BannerState(0, 119, 0, 0))
    assert live.f_succ[1] == pytest.approx(1.0)


def test_vacuum_adds_exact_success_mass_at_pull_30():
    without = ht_of(BannerSpec(BARE, 1, 40))
    with_v = ht_of(BannerSpec(replace(BARE, vacuum_at=30), 1, 40))
    survive = 1.0 - without.f_succ[:30].sum()
    p_vac = 1 - (1 - 0.004) ** 10
    assert with_v.f_succ[:30].tolist() == pytest.approx(without.f_succ[:30].tolist(), abs=1e-12)
    expected = without.f_succ[30] + (survive - without.f_succ[30]) * p_vac
    assert with_v.f_succ[30] == pytest.approx(expected, abs=1e-12)


def test_potential_at_240_completes_second_copy():
    rules = replace(BARE, potential_every=240)
    ht = ht_of(BannerSpec(rules, 2, 300), start=BannerState(0, 239, 1, 1))
    assert ht.f_succ[1] == pytest.approx(1.0)


def test_expected_pulls_monotone_in_entering_pity():
    """Spec §6: expected to hold; a failure is a research finding, not a bug to silence."""
    spec = BannerSpec(FULL, 1, 120)
    model = SingleBannerModel(spec)
    roots = [BannerState(t, 0, 0, 0) for t in range(FULL.hard_pity)]
    chain = EnumeratedChain.from_model(model, roots=roots)
    sv = state_values(chain)
    e = [sv.expectation[chain.index[r]] for r in roots]
    assert all(e[t + 1] <= e[t] + 1e-9 for t in range(len(e) - 1))


# frozen 2026-09-23, first UP, default rules, f=5, cap 120 (see docs/results.md, E2).
# Briefly moved on 2026-10-07 when R2 was set to a flat schedule, then restored the same day
# (docs/research-notes.md, Findings).
GOLDEN = {
    "mean": 74.3312129962629,
    "sd": 36.005262769041316,
    "q50": 67,
    "q90": 115,
    "q95": 115,
    "p_115": 0.32756723920235586,
}


def test_golden_default_rules():
    """Frozen after first computation (plan Task 8 step 5). Update only with a justification."""
    ht = ht_of(BannerSpec(FULL, 1, 120))
    assert rk.mean(ht) == pytest.approx(GOLDEN["mean"], rel=1e-9)
    assert rk.sd(ht) == pytest.approx(GOLDEN["sd"], rel=1e-9)
    assert rk.quantile(ht, 0.5) == GOLDEN["q50"]
    assert rk.quantile(ht, 0.9) == GOLDEN["q90"]
    assert rk.quantile(ht, 0.95) == GOLDEN["q95"]
    assert ht.f_succ[115] == pytest.approx(GOLDEN["p_115"], rel=1e-9)


def test_rerun_banner_has_bonus_pulls_at_60_and_90():
    from gacha.rules.endfield import rerun_rules

    rules = rerun_rules()
    p_vac = 1 - (1 - 0.004) ** 10
    for n in (59, 89):  # the counted pull that reaches 60 / 90 also grants 10 bonus pulls
        trs = pull(rules, BannerState(5, n, 0, 0), 0, 1)
        assert sum(p for p, nxt, _ in trs if nxt.c == 1) == pytest.approx(0.004 + 0.996 * p_vac)
    trs = pull(rules, BannerState(5, 70, 0, 0), 0, 1)  # no bonus between thresholds
    assert sum(p for p, nxt, _ in trs if nxt.c == 1) == pytest.approx(0.004)
    # resuming at n = 30, t = 30: the 120 guarantee is at most 90 pulls away
    # the cap counts every paid pull on the banner, including the 30 already made
    ht = ht_of(BannerSpec(rules, 1, 120), start=BannerState(30, 30, 0, 0))
    assert ht.horizon == 90 and ht.p_success == pytest.approx(1.0)
