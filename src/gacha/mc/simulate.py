"""Vectorized Monte Carlo on the enumerated chain; samples the same transitions the exact
engine uses, so rules are never re-implemented here (spec §5.7)."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

from gacha.kernel.chain import EnumeratedChain


@dataclass
class Paths:
    absorb: np.ndarray  # int8: 1 success, 2 fail
    paid: np.ndarray  # int64 paid pulls per path


def _tables(chain: EnumeratedChain) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dense per-state tables padded to the maximum out-degree.

    Padded cumulative probabilities are 1.0, so a uniform draw u < 1 never selects a pad."""
    n = chain.n
    deg = np.diff(chain.tr_ptr)
    width = int(deg.max())
    cum = np.ones((n, width))
    nxt = np.full((n, width), -1, dtype=np.int64)
    cost = np.zeros((n, width), dtype=np.int8)
    absorb = np.zeros((n, width), dtype=np.int8)
    ptr = chain.tr_ptr
    for i in range(n):
        a, b = int(ptr[i]), int(ptr[i + 1])
        k = b - a
        c = np.cumsum(chain.tr_prob[a:b])
        c[-1] = 1.0
        cum[i, :k] = c
        nxt[i, :k] = chain.tr_next[a:b]
        cost[i, :k] = chain.tr_cost[a:b]
        absorb[i, :k] = chain.tr_absorb[a:b]
    return cum, nxt, cost, absorb


def simulate(
    chain: EnumeratedChain, n_paths: int, seed: int, root: Hashable | None = None
) -> Paths:
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    if root is None:
        if len(chain.roots) != 1:
            raise ValueError("pass root explicitly when the chain has several roots")
        r = chain.roots[0]
    else:
        r = chain.index[root]
    cum, nxt, cost, absorb = _tables(chain)
    rng = np.random.default_rng(seed)
    state = np.full(n_paths, r, dtype=np.int64)
    paid = np.zeros(n_paths, dtype=np.int64)
    out = np.zeros(n_paths, dtype=np.int8)
    active = np.arange(n_paths)
    for _ in range(chain.n + 1):  # a DAG path visits each state at most once
        if active.size == 0:
            break
        s = state[active]
        u = rng.random(active.size)
        k = (cum[s] < u[:, None]).sum(axis=1)
        paid[active] += cost[s, k]
        ab = absorb[s, k]
        state[active] = nxt[s, k]
        done = ab > 0
        out[active[done]] = ab[done]
        active = active[~done]
    else:
        raise RuntimeError("simulation did not terminate; the chain should be a DAG")
    return Paths(out, paid)
