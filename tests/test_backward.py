import numpy as np
import pytest

from gacharisk.kernel.backward import state_values, success_within
from gacharisk.kernel.chain import EnumeratedChain
from gacharisk.kernel.forward import hitting_time
from tests.toy_models import ToyModel


def test_toy_state_values():
    chain = EnumeratedChain.from_model(ToyModel())
    sv = state_values(chain)
    a, b, c = (chain.index[s] for s in "abc")
    assert sv.expectation[[a, b, c]].tolist() == pytest.approx([1.2, 1.2, 0.5])
    assert sv.variance[[a, b, c]].tolist() == pytest.approx([0.16, 0.16, 0.25])
    assert sv.p_success[[a, b, c]].tolist() == pytest.approx([0.8, 0.8, 0.5])


def test_backward_matches_forward_moments():
    chain = EnumeratedChain.from_model(ToyModel())
    sv = state_values(chain)
    ht = hitting_time(chain)
    j = np.arange(ht.horizon + 1)
    pmf = ht.pmf_stop
    mean = float((j * pmf).sum())
    var = float(((j - mean) ** 2 * pmf).sum())
    root = chain.roots[0]
    assert sv.expectation[root] == pytest.approx(mean, abs=1e-12)
    assert sv.variance[root] == pytest.approx(var, abs=1e-12)
    assert sv.p_success[root] == pytest.approx(ht.p_success, abs=1e-12)


def test_success_within_toy():
    chain = EnumeratedChain.from_model(ToyModel())
    s = success_within(chain, 2)
    a, b, c = (chain.index[x] for x in "abc")
    assert s[:, c].tolist() == pytest.approx([0.0, 0.5, 0.5])
    assert s[:, b].tolist() == pytest.approx([0.0, 0.6, 0.8])
    assert s[:, a].tolist() == pytest.approx([0.0, 0.6, 0.8])


def test_success_within_matches_forward_cdf():
    chain = EnumeratedChain.from_model(ToyModel())
    ht = hitting_time(chain)
    s = success_within(chain, ht.horizon)
    assert s[:, chain.roots[0]].tolist() == pytest.approx(np.cumsum(ht.f_succ).tolist())
