import numpy as np
import pytest

from gacharisk.kernel.chain import EnumeratedChain
from gacharisk.kernel.forward import hitting_time
from tests.toy_models import ToyModel


def test_toy_hitting_time_exact():
    chain = EnumeratedChain.from_model(ToyModel())
    ht = hitting_time(chain)
    assert ht.f_succ.tolist() == pytest.approx([0.0, 0.6, 0.2])
    assert ht.f_fail.tolist() == pytest.approx([0.0, 0.2, 0.0])
    assert ht.residual == pytest.approx(0.0)
    assert ht.p_success == pytest.approx(0.8)


def test_mass_conservation():
    chain = EnumeratedChain.from_model(ToyModel())
    ht = hitting_time(chain)
    assert ht.f_succ.sum() + ht.f_fail.sum() + ht.residual == pytest.approx(1.0, abs=1e-12)


def test_truncated_horizon_reports_residual():
    chain = EnumeratedChain.from_model(ToyModel())
    ht = hitting_time(chain, horizon=1)
    assert ht.f_succ.tolist() == pytest.approx([0.0, 0.6])
    assert ht.f_fail.tolist() == pytest.approx([0.0, 0.2])
    assert ht.residual == pytest.approx(0.2)
    assert ht.residual_states == {"c": pytest.approx(0.2)}


def test_initial_distribution_over_states():
    chain = EnumeratedChain.from_model(ToyModel())
    mu0 = np.zeros(chain.n)
    mu0[chain.index["c"]] = 1.0
    ht = hitting_time(chain, mu0=mu0)
    assert ht.f_fail.tolist() == pytest.approx([0.5, 0.0, 0.0])
    assert ht.f_succ.tolist() == pytest.approx([0.0, 0.5, 0.0])


def test_rejects_bad_inputs():
    chain = EnumeratedChain.from_model(ToyModel())
    with pytest.raises(ValueError):
        hitting_time(chain, mu0=np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        hitting_time(chain, horizon=-1)
    multi = EnumeratedChain.from_model(ToyModel(), roots=["b", "c"])
    with pytest.raises(ValueError):
        hitting_time(multi)  # several roots: mu0 is required
