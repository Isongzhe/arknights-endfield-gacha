"""Forward propagation of probability mass over paid-pull budget levels (spec §5.2)."""

from __future__ import annotations

import numpy as np

from .chain import CycleError, EnumeratedChain
from .types import HittingTime


def hitting_time(
    chain: EnumeratedChain, mu0: np.ndarray | None = None, horizon: int | None = None
) -> HittingTime:
    """Distribution of paid pulls until absorption.

    For each budget level j: take the closure of free (cost-0) moves, record absorptions that
    happen through free transitions at level j and through paid transitions at level j+1, then
    advance the remaining mass. The default horizon is the exact support bound
    ``chain.max_cost_path``; a smaller horizon leaves a positive residual.
    """
    if mu0 is None:
        if len(chain.roots) != 1:
            raise ValueError("pass mu0 explicitly when the chain has several roots")
        mu = chain.point_mass(chain.states[chain.roots[0]])
    else:
        mu = np.array(mu0, dtype=float)
        if mu.shape != (chain.n,) or (mu < 0).any() or abs(mu.sum() - 1.0) > 1e-9:
            raise ValueError("mu0 must be a probability vector over chain.states")
    horizon_j = chain.max_cost_path if horizon is None else int(horizon)
    if horizon_j < 0:
        raise ValueError("horizon must be >= 0")

    f_succ = np.zeros(horizon_j + 1)
    f_fail = np.zeros(horizon_j + 1)
    residual_measure = np.zeros(chain.n)
    for j in range(horizon_j + 1):
        w = _free_closure(chain, mu)
        f_succ[j] += w @ chain.a0_succ
        f_fail[j] += w @ chain.a0_fail
        if j < horizon_j:
            f_succ[j + 1] += w @ chain.a1_succ
            f_fail[j + 1] += w @ chain.a1_fail
            mu = chain.Q1T @ w
        else:
            residual_measure = w * chain.paid_out
    residual = float(residual_measure.sum())
    residual_states = {
        chain.states[i]: float(residual_measure[i]) for i in np.nonzero(residual_measure > 0)[0]
    }
    return HittingTime(f_succ, f_fail, residual, residual_states)


def _free_closure(chain: EnumeratedChain, mu: np.ndarray) -> np.ndarray:
    """W = sum_{i>=0} mu Q0^i; terminates because cost-0 transitions form a DAG."""
    w = mu.copy()
    v = mu
    for _ in range(chain.n + 1):
        v = chain.Q0T @ v
        if not v.any():
            return w
        w = w + v
    raise CycleError("free-transition closure did not terminate")
