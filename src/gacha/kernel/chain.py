"""Enumerate a model's reachable states into arrays and sparse matrices."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .types import FAIL, SUCCESS, Model

ABSORB_CODE = {None: 0, SUCCESS: 1, FAIL: 2}
PROB_TOL = 1e-9


class CycleError(ValueError):
    """The state graph has a cycle; models must strictly advance a counter on every step."""


@dataclass
class EnumeratedChain:
    states: list[Hashable]
    index: dict[Hashable, int]
    roots: list[int]
    tr_ptr: np.ndarray  # (n+1,) CSR row pointers into the transition arrays
    tr_prob: np.ndarray  # (m,) float
    tr_next: np.ndarray  # (m,) int64, -1 when absorbing
    tr_cost: np.ndarray  # (m,) int8, 0 or 1
    tr_absorb: np.ndarray  # (m,) int8, 0 none / 1 success / 2 fail
    Q0T: sparse.csr_matrix  # transpose of the cost-0 state->state matrix
    Q1T: sparse.csr_matrix  # transpose of the cost-1 state->state matrix
    a0_succ: np.ndarray
    a0_fail: np.ndarray
    a1_succ: np.ndarray
    a1_fail: np.ndarray
    paid_out: np.ndarray  # (n,) total probability of cost-1 transitions out of each state
    topo: np.ndarray  # topological order (sources first)
    max_cost_path: int  # longest cost-weighted path from any root = support bound of T

    @property
    def n(self) -> int:
        return len(self.states)

    def point_mass(self, state: Hashable) -> np.ndarray:
        v = np.zeros(self.n)
        v[self.index[state]] = 1.0
        return v

    @classmethod
    def from_model(cls, model: Model, roots: Sequence[Hashable] | None = None) -> EnumeratedChain:
        root_states = [model.initial()] if roots is None else list(roots)
        if not root_states:
            raise ValueError("at least one root state is required")
        states: list[Hashable] = []
        index: dict[Hashable, int] = {}
        queue: deque[Hashable] = deque()
        for s in root_states:
            if s not in index:
                index[s] = len(states)
                states.append(s)
                queue.append(s)
        root_idx = [index[s] for s in root_states]

        # BFS: states are appended at discovery and popped FIFO, so rows[i] belongs to states[i].
        rows: list[list[tuple[float, int, int, int]]] = []
        while queue:
            s = queue.popleft()
            trs = model.step(s)
            total = sum(tr.prob for tr in trs)
            if abs(total - 1.0) > PROB_TOL:
                raise ValueError(f"transition probabilities from {s!r} sum to {total!r}, not 1")
            merged: dict[tuple[Hashable | None, int, str | None], float] = {}
            for tr in trs:
                if tr.prob == 0.0:
                    continue
                key = (tr.next, tr.cost, tr.absorb)
                merged[key] = merged.get(key, 0.0) + tr.prob
            row = []
            for (nxt, cost, absorb), p in merged.items():
                if nxt is None:
                    j = -1
                else:
                    if nxt not in index:
                        index[nxt] = len(states)
                        states.append(nxt)
                        queue.append(nxt)
                    j = index[nxt]
                row.append((p, j, cost, ABSORB_CODE[absorb]))
            rows.append(row)

        n = len(states)
        counts = np.array([len(r) for r in rows], dtype=np.int64)
        tr_ptr = np.zeros(n + 1, dtype=np.int64)
        tr_ptr[1:] = np.cumsum(counts)
        flat = [tr for r in rows for tr in r]
        tr_prob = np.array([f[0] for f in flat], dtype=float)
        tr_next = np.array([f[1] for f in flat], dtype=np.int64)
        tr_cost = np.array([f[2] for f in flat], dtype=np.int8)
        tr_absorb = np.array([f[3] for f in flat], dtype=np.int8)

        topo = _topological_order(n, tr_ptr, tr_next)
        max_cost_path = _max_cost_path(topo, tr_ptr, tr_next, tr_cost, root_idx)

        src = np.repeat(np.arange(n, dtype=np.int64), counts)
        to_state = tr_next >= 0
        m0 = (tr_cost == 0) & to_state
        m1 = (tr_cost == 1) & to_state
        q0 = sparse.csr_matrix((tr_prob[m0], (src[m0], tr_next[m0])), shape=(n, n))
        q1 = sparse.csr_matrix((tr_prob[m1], (src[m1], tr_next[m1])), shape=(n, n))

        def absorb_vector(cost: int, code: int) -> np.ndarray:
            mask = (tr_cost == cost) & (tr_absorb == code)
            v = np.zeros(n)
            np.add.at(v, src[mask], tr_prob[mask])
            return v

        paid_out = np.zeros(n)
        np.add.at(paid_out, src[tr_cost == 1], tr_prob[tr_cost == 1])

        return cls(
            states=states,
            index=index,
            roots=root_idx,
            tr_ptr=tr_ptr,
            tr_prob=tr_prob,
            tr_next=tr_next,
            tr_cost=tr_cost,
            tr_absorb=tr_absorb,
            Q0T=q0.transpose().tocsr(),
            Q1T=q1.transpose().tocsr(),
            a0_succ=absorb_vector(0, 1),
            a0_fail=absorb_vector(0, 2),
            a1_succ=absorb_vector(1, 1),
            a1_fail=absorb_vector(1, 2),
            paid_out=paid_out,
            topo=topo,
            max_cost_path=max_cost_path,
        )


def _topological_order(n: int, tr_ptr: np.ndarray, tr_next: np.ndarray) -> np.ndarray:
    ptr = tr_ptr.tolist()
    nxt = tr_next.tolist()
    indeg = [0] * n
    for j in nxt:
        if j >= 0:
            indeg[j] += 1
    ready = deque(i for i in range(n) if indeg[i] == 0)
    order: list[int] = []
    while ready:
        i = ready.popleft()
        order.append(i)
        for k in range(ptr[i], ptr[i + 1]):
            j = nxt[k]
            if j >= 0:
                indeg[j] -= 1
                if indeg[j] == 0:
                    ready.append(j)
    if len(order) != n:
        raise CycleError(
            "state graph has a cycle; every transition must strictly increase a counter"
        )
    return np.array(order, dtype=np.int64)


def _max_cost_path(
    topo: np.ndarray,
    tr_ptr: np.ndarray,
    tr_next: np.ndarray,
    tr_cost: np.ndarray,
    roots: list[int],
) -> int:
    ptr = tr_ptr.tolist()
    nxt = tr_next.tolist()
    cost = tr_cost.tolist()
    longest = [0] * len(topo)
    for i in reversed(topo.tolist()):
        best = 0
        for k in range(ptr[i], ptr[i + 1]):
            j = nxt[k]
            v = cost[k] + (longest[j] if j >= 0 else 0)
            if v > best:
                best = v
        longest[i] = best
    return max(longest[r] for r in roots)
