import numpy as np
import pytest

from gacharisk.analysis.rebate import effective_stock, quota_per_pull, star_rates
from gacharisk.rules.endfield import EndfieldCharacterRules

RULES = EndfieldCharacterRules()


def test_six_star_rate_matches_the_pity_cycle():
    rate6, _ = star_rates(RULES)
    p = np.array(RULES.probs)
    cycle = np.concatenate([[1.0], np.cumprod(1 - p[:-1])]).sum()
    assert rate6 == pytest.approx(1 / cycle, abs=1e-9)  # 1/53.8993, the official 1.8553%


def test_five_star_rate_against_simulation():
    _, rate5 = star_rates(RULES)
    rng = np.random.default_rng(7)
    p = RULES.probs
    t = g = five = 0
    n = 400_000
    for u in rng.random(n):
        p6 = p[t]
        p5 = 1 - p6 if g == 9 else min(RULES.five_star_rate, 1 - p6)
        if u < p6:
            t = g = 0
        elif u < p6 + p5:
            five += 1
            t, g = t + 1, 0
        else:
            t, g = t + 1, g + 1
    assert rate5 == pytest.approx(five / n, abs=0.002)
    assert 0.11 < rate5 < 0.14  # the 10-pull guarantee lifts it well above the 8% base


def test_quota_per_pull_and_effective_stock():
    rate6, rate5 = star_rates(RULES)
    assert quota_per_pull(RULES, owned_five_share=0.0, owned_offrate_share=0.0) == 0.0
    full5 = quota_per_pull(RULES, owned_five_share=1.0, owned_offrate_share=0.0)
    assert full5 == pytest.approx(10 * rate5)
    both = quota_per_pull(RULES, owned_five_share=1.0, owned_offrate_share=1.0)
    assert both == pytest.approx(10 * rate5 + 50 * rate6 * (1 - RULES.up_share))
    assert effective_stock(89, quota_now=0, per_pull=0.0) == 89
    assert effective_stock(89, quota_now=25, per_pull=0.0) == 90
    assert effective_stock(89, quota_now=10, per_pull=full5) == int((89 + 0.4) / (1 - full5 / 25))
    with pytest.raises(ValueError):
        quota_per_pull(RULES, owned_five_share=1.2, owned_offrate_share=0.0)
