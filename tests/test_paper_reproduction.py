"""Regression tests against Hou, Zhu & Zhang, Symmetry 2026, 18(6), 1051.

Tolerances (spec §6): 0.01 on moments, exact on integer quantiles, 5e-4 on probabilities.
A disagreement is investigated and documented in docs/research-notes.md, never forced.
"""

import pytest

from gacha.kernel.backward import state_values
from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.models.paper import Featured5050Model, IIDStagesModel, SingleCounterModel
from gacha.risk import metrics as rk
from gacha.risk.normal import max_abs_cdf_error
from gacha.rules.paper import PaperSchedule

MOM = 0.01
PROB = 5e-4


@pytest.fixture(scope="module")
def soft():
    chain = EnumeratedChain.from_model(SingleCounterModel(PaperSchedule().probs))
    return chain, hitting_time(chain)


@pytest.fixture(scope="module")
def t10(soft):
    return rk.from_pmf(rk.convolve(soft[1].pmf_stop, 10))


def test_single_stage_moments(soft):
    _, ht = soft
    assert ht.p_success == pytest.approx(1.0, abs=1e-12)
    assert rk.mean(ht) == pytest.approx(62.34, abs=MOM)
    assert rk.var(ht) == pytest.approx(592.43, abs=MOM)
    assert rk.sd(ht) == pytest.approx(24.34, abs=MOM)
    assert ht.horizon == 90


def test_expected_remaining_table(soft):
    chain, _ = soft
    sv = state_values(chain)
    expected = {0: 62.34, 70: 7.68, 72: 5.76, 73: 4.78, 80: 1.92, 89: 1.00}
    for t, value in expected.items():
        assert sv.expectation[chain.index[(t,)]] == pytest.approx(value, abs=MOM)


def test_variance_recurrence_matches_paper_form(soft):
    chain, _ = soft
    sv = state_values(chain)
    p = PaperSchedule().probs
    e = [sv.expectation[chain.index[(t,)]] for t in range(90)]
    v = [sv.variance[chain.index[(t,)]] for t in range(90)]
    for t in range(89):  # paper Eq. (10)
        expected = (1 - p[t]) * v[t + 1] + p[t] * (1 - p[t]) * e[t + 1] ** 2
        assert v[t] == pytest.approx(expected, abs=1e-9)
    assert v[89] == pytest.approx(0.0, abs=1e-12)


def test_hard_pity_only_baseline():
    model = SingleCounterModel(PaperSchedule().hard_only().probs)
    ht = hitting_time(EnumeratedChain.from_model(model))
    assert rk.mean(ht) == pytest.approx(69.70, abs=MOM)
    assert rk.skewness(ht) == pytest.approx(-1.08, abs=MOM)
    assert rk.entropy_nats(ht) == pytest.approx(2.54, abs=MOM)


def test_single_stage_asymmetry(soft):
    _, ht = soft
    assert rk.skewness(ht) == pytest.approx(-1.24, abs=MOM)
    assert rk.entropy_nats(ht) == pytest.approx(3.61, abs=MOM)


def test_t10_moments_and_quantiles(t10):
    assert rk.mean(t10) == pytest.approx(623.38, abs=MOM)
    assert rk.var(t10) == pytest.approx(5924.34, abs=MOM)
    assert [rk.quantile(t10, a) for a in (0.5, 0.75, 0.9, 0.95, 0.99)] == [629, 679, 719, 741, 775]


def test_t10_tail_and_budget_metrics(t10):
    assert rk.survival(t10, 724) == pytest.approx(0.0876, abs=PROB)
    assert rk.survival(t10, 778) == pytest.approx(0.0073, abs=PROB)
    assert rk.cvar(t10, 0.9) == pytest.approx(744.66, abs=MOM)
    assert rk.expected_excess(t10, 700) == pytest.approx(5.11, abs=MOM)
    assert rk.completion(t10, 719) == pytest.approx(0.9014, abs=PROB)


def test_t10_via_iid_stages_model_equals_convolution(t10):
    model = IIDStagesModel(SingleCounterModel(PaperSchedule().probs), 10)
    ht = hitting_time(EnumeratedChain.from_model(model))
    assert ht.pmf_stop.tolist() == pytest.approx(t10.pmf_stop.tolist(), abs=1e-12)


def test_normal_approximation_error(t10):
    err = max_abs_cdf_error(t10)
    assert 0.02 < err < 0.04  # paper: "approximately 0.03"


def test_featured_target_no_guarantee():
    ht = hitting_time(EnumeratedChain.from_model(Featured5050Model(PaperSchedule().probs)))
    assert ht.horizon == 180
    assert rk.mean(ht) == pytest.approx(93.51, abs=MOM)
    assert rk.sd(ht) == pytest.approx(43.13, abs=MOM)
    assert [rk.quantile(ht, a) for a in (0.9, 0.95, 0.99)] == [156, 158, 161]
    assert rk.skewness(ht) == pytest.approx(0.01, abs=MOM)
    assert rk.entropy_nats(ht) == pytest.approx(4.54, abs=MOM)


def test_featured_target_with_guarantee_equals_single_stage(soft):
    model = Featured5050Model(PaperSchedule().probs, guarantee=True)
    ht = hitting_time(EnumeratedChain.from_model(model))
    assert ht.pmf_stop.tolist() == pytest.approx(soft[1].pmf_stop.tolist(), abs=1e-12)
    assert [rk.quantile(ht, a) for a in (0.9, 0.95, 0.99)] == [80, 81, 83]


@pytest.mark.parametrize(
    "m, mean, sd, q90, q95",
    [
        (2, 187.01, 60.99, 265, 300),
        (5, 467.53, 96.44, 593, 626),
        (10, 935.06, 136.39, 1111, 1160),
    ],
)
def test_repeated_featured_targets(m, mean, sd, q90, q95):
    model = IIDStagesModel(Featured5050Model(PaperSchedule().probs), m)
    ht = hitting_time(EnumeratedChain.from_model(model))
    assert rk.mean(ht) == pytest.approx(mean, abs=MOM)
    assert rk.sd(ht) == pytest.approx(sd, abs=MOM)
    assert rk.quantile(ht, 0.9) == q90
    assert rk.quantile(ht, 0.95) == q95
