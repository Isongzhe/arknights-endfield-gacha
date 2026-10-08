import pytest

from gacharisk.cost.menu import ENDFIELD_TW_STANDARD, PriceMenu


def test_min_cost_matches_hand_checked_values():
    m = ENDFIELD_TW_STANDARD
    assert m.min_cost(0) == 0
    assert m.min_cost(-3) == 0
    assert m.min_cost(10) == 490  # the banner pack: 10 pulls for NT$490
    assert m.min_cost(27, leftover_jade=475) == 1480
    assert m.min_cost(116, leftover_jade=475) == 6420
    assert m.min_cost(127, leftover_jade=175) == 7000


def test_pack_can_be_disabled_and_cost_is_monotone():
    m = ENDFIELD_TW_STANDARD
    assert m.min_cost(10, use_packs=False) > 490
    costs = [m.min_cost(k) for k in range(0, 130)]
    assert all(b >= a for a, b in zip(costs, costs[1:], strict=False))


def test_max_pulls_for_budget():
    m = ENDFIELD_TW_STANDARD
    assert m.max_pulls(0) == 0
    assert m.max_pulls(490) == 10
    k = m.max_pulls(2000, leftover_jade=475)
    assert m.min_cost(k, leftover_jade=475) <= 2000 < m.min_cost(k + 1, leftover_jade=475)


def test_menu_validation():
    with pytest.raises(ValueError):
        PriceMenu(stone_tiers=())
    with pytest.raises(ValueError):
        PriceMenu(stone_tiers=((100, 0),))
