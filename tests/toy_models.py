"""Tiny hand-checkable models used by the kernel tests."""

from gacha.kernel.types import FAIL, SUCCESS, Transition


class ToyModel:
    """a -(free)-> b; b: success paid 0.6 / c paid 0.4; c: fail free 0.5 / success paid 0.5.

    Hand results: f_succ = [0, 0.6, 0.2], f_fail = [0, 0.2, 0]; P(success) = 0.8;
    E[T_stop] = 1.2, Var = 0.16; E[b] = 1.2, E[c] = 0.5, Var[c] = 0.25; max_cost_path = 2.
    """

    def initial(self):
        return "a"

    def step(self, s):
        if s == "a":
            return [Transition(1.0, "b", cost=0)]
        if s == "b":
            return [Transition(0.6, None, cost=1, absorb=SUCCESS), Transition(0.4, "c", cost=1)]
        if s == "c":
            return [
                Transition(0.5, None, cost=0, absorb=FAIL),
                Transition(0.5, None, cost=1, absorb=SUCCESS),
            ]
        raise KeyError(s)


class CyclicModel:
    def initial(self):
        return "x"

    def step(self, s):
        if s == "x":
            return [Transition(0.5, "y", cost=1), Transition(0.5, None, cost=1, absorb=SUCCESS)]
        return [Transition(1.0, "x", cost=1)]


class BadProbModel:
    def initial(self):
        return 0

    def step(self, s):
        return [Transition(0.7, None, cost=1, absorb=SUCCESS)]


class DuplicateTargetModel:
    """Two transitions to the same (next, cost, absorb) must be merged by the chain."""

    def initial(self):
        return 0

    def step(self, s):
        if s == 0:
            return [
                Transition(0.25, 1, cost=1),
                Transition(0.25, 1, cost=1),
                Transition(0.5, None, cost=1, absorb=SUCCESS),
            ]
        return [Transition(1.0, None, cost=1, absorb=SUCCESS)]
