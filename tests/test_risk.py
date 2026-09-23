import math

import numpy as np
import pytest

from gacha.kernel.types import HittingTime
from gacha.risk import metrics as rk
from gacha.risk.normal import max_abs_cdf_error, normal_approx_cdf


@pytest.fixture
def ht():
    # T in {1, 2, 3} with probabilities 0.5, 0.3, 0.2 (all success)
    return HittingTime(np.array([0.0, 0.5, 0.3, 0.2]), np.zeros(4), 0.0)


def test_moments(ht):
    assert rk.mean(ht) == pytest.approx(1.7)
    assert rk.var(ht) == pytest.approx(0.61)
    assert rk.sd(ht) == pytest.approx(math.sqrt(0.61))
    assert rk.cv(ht) == pytest.approx(math.sqrt(0.61) / 1.7)
    assert rk.skewness(ht) == pytest.approx(0.2760 / 0.61**1.5, abs=1e-3)
    assert rk.entropy_nats(ht) == pytest.approx(1.0297, abs=1e-4)


def test_cdf_survival_quantiles(ht):
    assert rk.cdf(ht, -1) == 0.0
    assert rk.cdf(ht, 1) == pytest.approx(0.5)
    assert rk.cdf(ht, 2) == pytest.approx(0.8)
    assert rk.cdf(ht, 99) == pytest.approx(1.0)
    assert rk.survival(ht, 2) == pytest.approx(0.5)
    assert rk.survival(ht, 0) == pytest.approx(1.0)
    assert rk.quantile(ht, 0.5) == 1
    assert rk.quantile(ht, 0.8) == 2
    assert rk.quantile(ht, 0.81) == 3
    assert rk.var_at(ht, 0.8) == 2
    with pytest.raises(ValueError):
        rk.quantile(ht, 0.0)
    with pytest.raises(ValueError):
        rk.quantile(ht, 1.5)


def test_cvar_and_expected_excess(ht):
    assert rk.cvar(ht, 0.8) == pytest.approx(2.4)
    assert rk.expected_excess(ht, 1) == pytest.approx(0.7)
    assert rk.expected_excess(ht, 3) == pytest.approx(0.0)


def test_defective_distribution_conventions():
    ht = HittingTime(np.array([0.0, 0.6, 0.2]), np.array([0.0, 0.2, 0.0]), 0.0)
    assert rk.mean(ht) == pytest.approx(1.2)  # stop distribution, mass 1
    assert rk.mean(ht, which="success") == pytest.approx(1.25)  # conditional on success
    assert rk.completion(ht, 1) == pytest.approx(0.6)
    assert rk.completion(ht, 2) == pytest.approx(0.8)
    assert rk.completion(ht, -3) == 0.0
    assert rk.cdf(ht, 1, which="success") == pytest.approx(0.75)


def test_truncated_stop_distribution_rejected():
    ht = HittingTime(np.array([0.0, 0.6]), np.array([0.0, 0.2]), 0.2)
    with pytest.raises(ValueError):
        rk.mean(ht)
    assert rk.mean(ht, which="success") == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rk.pmf_of(ht, "nonsense")


def test_convolve_and_from_pmf():
    assert rk.convolve(np.array([0.0, 0.5, 0.5]), 2).tolist() == pytest.approx(
        [0, 0, 0.25, 0.5, 0.25]
    )
    assert rk.convolve(np.array([0.0, 1.0]), 1).tolist() == pytest.approx([0.0, 1.0])
    h = rk.from_pmf(np.array([0.0, 0.25, 0.75]))
    assert h.p_success == pytest.approx(1.0)
    assert rk.mean(h) == pytest.approx(1.75)


def test_summary_dataframe(ht):
    df = rk.summary(ht, alphas=(0.5, 0.9), budgets=(1, 2))
    assert set(df.columns) == {"metric", "value"}
    rows = dict(zip(df["metric"], df["value"], strict=True))
    assert rows["mean"] == pytest.approx(1.7)
    assert rows["q50"] == 1
    assert rows["completion_at_2"] == pytest.approx(0.8)
    assert rows["expected_excess_at_1"] == pytest.approx(0.7)


def test_normal_approximation_on_binomial():
    pmf = rk.convolve(np.array([0.5, 0.5]), 40)  # Binomial(40, 0.5)
    h = rk.from_pmf(pmf)
    assert normal_approx_cdf(h, 20) == pytest.approx(rk.cdf(h, 20), abs=0.01)
    assert max_abs_cdf_error(h) < 0.01
