"""Backward (first-step) recurrences on the enumerated DAG (spec §5.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chain import EnumeratedChain


@dataclass
class StateValues:
    expectation: np.ndarray  # E[paid pulls until absorption | state]
    variance: np.ndarray
    p_success: np.ndarray


def state_values(chain: EnumeratedChain) -> StateValues:
    """E, Var and P(success) per state via reverse topological order.

    E[s]  = sum p (cost + E[next]);  M2[s] = sum p (cost^2 + 2 cost E[next] + M2[next]);
    Var = M2 - E^2;  P[s] = sum p (1{absorb=success} + P[next]).  Absorbed values are 0.
    """
    n = chain.n
    ptr = chain.tr_ptr.tolist()
    prob = chain.tr_prob.tolist()
    nxt = chain.tr_next.tolist()
    cost = chain.tr_cost.tolist()
    absorb = chain.tr_absorb.tolist()
    e = [0.0] * n
    m2 = [0.0] * n
    ps = [0.0] * n
    for i in reversed(chain.topo.tolist()):
        ei = m2i = pi = 0.0
        for k in range(ptr[i], ptr[i + 1]):
            p = prob[k]
            j = nxt[k]
            c = cost[k]
            if j >= 0:
                en, m2n, pn = e[j], m2[j], ps[j]
            else:
                en = m2n = 0.0
                pn = 1.0 if absorb[k] == 1 else 0.0
            ei += p * (c + en)
            m2i += p * (c * c + 2.0 * c * en + m2n)
            pi += p * pn
        e[i], m2[i], ps[i] = ei, m2i, pi
    expectation = np.array(e)
    variance = np.array(m2) - expectation**2
    return StateValues(expectation, variance, np.array(ps))


def success_within(chain: EnumeratedChain, max_budget: int) -> np.ndarray:
    """S[b, i] = P(success using at most b paid pulls | start in state i), b = 0..max_budget."""
    if max_budget < 0:
        raise ValueError("max_budget must be >= 0")
    n = chain.n
    ptr = chain.tr_ptr.tolist()
    prob = chain.tr_prob.tolist()
    nxt = chain.tr_next.tolist()
    cost = chain.tr_cost.tolist()
    absorb = chain.tr_absorb.tolist()
    s = np.zeros((max_budget + 1, n))
    for i in reversed(chain.topo.tolist()):
        acc = np.zeros(max_budget + 1)
        for k in range(ptr[i], ptr[i + 1]):
            p = prob[k]
            j = nxt[k]
            if j < 0:
                if absorb[k] == 1:
                    if cost[k] == 0:
                        acc += p
                    else:
                        acc[1:] += p
            elif cost[k] == 0:
                acc += p * s[:, j]
            else:
                acc[1:] += p * s[:-1, j]
        s[:, i] = acc
    return s
