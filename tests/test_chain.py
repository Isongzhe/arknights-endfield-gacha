import numpy as np
import pytest

from gacha.kernel.chain import CycleError, EnumeratedChain
from gacha.kernel.types import SUCCESS, HittingTime, Transition
from tests.toy_models import BadProbModel, CyclicModel, DuplicateTargetModel, ToyModel


def test_transition_validation():
    with pytest.raises(ValueError):
        Transition(0.5, None, cost=1)  # absorbing needs a label
    with pytest.raises(ValueError):
        Transition(0.5, "s", cost=1, absorb=SUCCESS)  # non-absorbing must not carry a label
    with pytest.raises(ValueError):
        Transition(0.5, "s", cost=2)
    with pytest.raises(ValueError):
        Transition(-0.1, "s")
    with pytest.raises(ValueError):
        Transition(0.5, None, absorb="other")


def test_hitting_time_properties():
    ht = HittingTime(np.array([0.0, 0.6, 0.2]), np.array([0.0, 0.2, 0.0]), 0.0)
    assert ht.horizon == 2
    assert ht.p_success == pytest.approx(0.8)
    assert ht.p_fail == pytest.approx(0.2)
    assert ht.pmf_stop.tolist() == pytest.approx([0.0, 0.8, 0.2])


def test_enumerates_toy_model():
    chain = EnumeratedChain.from_model(ToyModel())
    assert chain.n == 3
    assert chain.states[chain.roots[0]] == "a"
    assert set(chain.states) == {"a", "b", "c"}
    assert chain.max_cost_path == 2
    # a comes before b before c in topological order
    pos = {chain.states[i]: k for k, i in enumerate(chain.topo.tolist())}
    assert pos["a"] < pos["b"] < pos["c"]


def test_transition_arrays_and_vectors():
    chain = EnumeratedChain.from_model(ToyModel())
    b, c = chain.index["b"], chain.index["c"]
    assert chain.a1_succ[b] == pytest.approx(0.6)
    assert chain.a0_fail[c] == pytest.approx(0.5)
    assert chain.a1_succ[c] == pytest.approx(0.5)
    assert chain.paid_out[b] == pytest.approx(1.0)
    assert chain.paid_out[c] == pytest.approx(0.5)
    assert chain.paid_out[chain.index["a"]] == pytest.approx(0.0)
    # Q0T[b, a] = 1 (a -> b free); Q1T[c, b] = 0.4 (b -> c paid)
    assert chain.Q0T[b, chain.index["a"]] == pytest.approx(1.0)
    assert chain.Q1T[c, b] == pytest.approx(0.4)
    v = chain.point_mass("b")
    assert v.sum() == 1.0 and v[b] == 1.0


def test_merges_duplicate_targets():
    chain = EnumeratedChain.from_model(DuplicateTargetModel())
    root = chain.roots[0]
    k = slice(chain.tr_ptr[root], chain.tr_ptr[root + 1])
    assert chain.tr_prob[k].size == 2
    assert sorted(chain.tr_prob[k].tolist()) == pytest.approx([0.5, 0.5])


def test_cycle_detection():
    with pytest.raises(CycleError):
        EnumeratedChain.from_model(CyclicModel())


def test_probability_validation():
    with pytest.raises(ValueError):
        EnumeratedChain.from_model(BadProbModel())


def test_multiple_roots():
    chain = EnumeratedChain.from_model(ToyModel(), roots=["b", "c"])
    assert set(chain.states) == {"b", "c"}
    assert len(chain.roots) == 2
    assert chain.max_cost_path == 2
