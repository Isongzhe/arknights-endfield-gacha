import numpy as np
import pytest

from gacha.kernel.chain import EnumeratedChain
from gacha.kernel.forward import hitting_time
from gacha.mc.compare import compare
from gacha.mc.simulate import simulate
from gacha.models.endfield import SingleBannerModel
from gacha.models.paper import SingleCounterModel
from gacha.models.plan import Plan
from gacha.rules.endfield import BannerSpec, EndfieldCharacterRules
from gacha.rules.paper import PaperSchedule
from tests.toy_models import ToyModel

N = 200_000
SEED = 20260923


def test_toy_paths_have_expected_shape_and_frequencies():
    chain = EnumeratedChain.from_model(ToyModel())
    paths = simulate(chain, 50_000, seed=1)
    assert paths.absorb.shape == (50_000,) and paths.paid.shape == (50_000,)
    assert set(np.unique(paths.absorb).tolist()) <= {1, 2}
    assert (paths.absorb == 1).mean() == pytest.approx(0.8, abs=0.01)
    assert paths.paid.mean() == pytest.approx(1.2, abs=0.01)


def test_simulation_is_reproducible():
    chain = EnumeratedChain.from_model(ToyModel())
    a = simulate(chain, 1000, seed=7)
    b = simulate(chain, 1000, seed=7)
    assert np.array_equal(a.paid, b.paid) and np.array_equal(a.absorb, b.absorb)


def test_compare_flags_agreement():
    chain = EnumeratedChain.from_model(ToyModel())
    df = compare(hitting_time(chain), simulate(chain, N, seed=SEED), thresholds=(1, 2))
    assert set(df["metric"]) == {"p_success", "mean", "survival_at_1", "survival_at_2"}
    assert df["ok"].all()


def _parity(chain, thresholds):
    df = compare(hitting_time(chain), simulate(chain, N, seed=SEED), thresholds=thresholds)
    assert df["ok"].all(), df.to_string()


def test_parity_paper_single_counter():
    _parity(EnumeratedChain.from_model(SingleCounterModel(PaperSchedule().probs)), (80,))


def test_parity_endfield_first_up():
    spec = BannerSpec(EndfieldCharacterRules(), 1, 120)
    _parity(EnumeratedChain.from_model(SingleBannerModel(spec)), (60, 100))


def test_parity_two_banner_plan():
    spec = BannerSpec(EndfieldCharacterRules(), 1, 120)
    _parity(EnumeratedChain.from_model(Plan([spec, spec])), (120, 180))


def test_simulate_requires_root_for_multi_root_chain():
    chain = EnumeratedChain.from_model(ToyModel(), roots=["b", "c"])
    with pytest.raises(ValueError):
        simulate(chain, 10, seed=0)
    paths = simulate(chain, 10, seed=0, root="c")
    assert paths.paid.max() <= 1
