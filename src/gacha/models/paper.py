"""Models from Hou, Zhu & Zhang (2026): single pity counter, featured 50/50, i.i.d. stages."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

from gacha.kernel.types import FAIL, SUCCESS, Model, Transition


class SingleCounterModel:
    """State (t,), t = consecutive failures. Success with p_t, else t+1. Paper §2-3."""

    def __init__(self, probs: Sequence[float]):
        self.p = tuple(float(x) for x in probs)
        if not self.p or self.p[-1] != 1.0:
            raise ValueError("the last probability must be 1 (hard pity)")

    def initial(self) -> Hashable:
        return (0,)

    def step(self, s: Hashable) -> list[Transition]:
        (t,) = s
        p = self.p[t]
        out = [Transition(p, None, cost=1, absorb=SUCCESS)]
        if p < 1.0:
            out.append(Transition(1.0 - p, (t + 1,), cost=1))
        return out


class Featured5050Model:
    """State (t, g): g=1 means the next rare item is guaranteed featured. Paper Eq. (21)-(22)."""

    def __init__(self, probs: Sequence[float], q: float = 0.5, guarantee: bool = False):
        self.p = tuple(float(x) for x in probs)
        if not self.p or self.p[-1] != 1.0:
            raise ValueError("the last probability must be 1 (hard pity)")
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be in [0, 1]")
        self.q = q
        self.guarantee = guarantee

    def initial(self) -> Hashable:
        return (0, 1 if self.guarantee else 0)

    def step(self, s: Hashable) -> list[Transition]:
        t, g = s
        p = self.p[t]
        out: list[Transition] = []
        if g == 1:
            out.append(Transition(p, None, cost=1, absorb=SUCCESS))
        else:
            out.append(Transition(p * self.q, None, cost=1, absorb=SUCCESS))
            out.append(Transition(p * (1.0 - self.q), (0, 1), cost=1))
        if p < 1.0:
            out.append(Transition(1.0 - p, (t + 1, g), cost=1))
        return out


class IIDStagesModel:
    """m independent repetitions of ``model``; each success restarts from model.initial()."""

    def __init__(self, model: Model, m: int):
        if m < 1:
            raise ValueError("m must be >= 1")
        self.model = model
        self.m = m

    def initial(self) -> Hashable:
        return (0, self.model.initial())

    def step(self, s: Hashable) -> list[Transition]:
        stage, inner = s
        out: list[Transition] = []
        for tr in self.model.step(inner):
            if tr.absorb == SUCCESS:
                if stage + 1 == self.m:
                    out.append(Transition(tr.prob, None, cost=tr.cost, absorb=SUCCESS))
                else:
                    nxt = (stage + 1, self.model.initial())
                    out.append(Transition(tr.prob, nxt, cost=tr.cost))
            elif tr.absorb == FAIL:
                out.append(tr)
            else:
                out.append(Transition(tr.prob, (stage, tr.next), cost=tr.cost))
        return out
