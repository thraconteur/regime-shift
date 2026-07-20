"""
optimizer.py — Dynamic Constraint Mapping
============================================
Maps the CURRENT regime to a different convex optimization objective, then
solves it with CVXPY. All portfolios are long-only (w >= 0) and fully
invested (sum(w) == 1) — a fresher-friendly, leverage-free constraint set
that's also what most real allocation mandates require.

Regime -> objective:
    Bull    -> Maximum Sharpe Ratio  (go for growth, equities favoured)
    Bear    -> Minimum Variance      (capital preservation, de-risk)
    Crisis  -> Minimum Variance + a minimum floor allocation to Gold
               (explicit flight-to-safety; min-vol alone can still leave
               you overweight bonds if bond vol is temporarily low, so we
               force some crisis-hedge exposure directly)

Max-Sharpe convex reformulation
--------------------------------
Sharpe = (mu^T w) / sqrt(w^T Sigma w) is NOT convex in w. The standard
long-only-friendly trick (Cornuejols & Tütüncü) is a change of variables:
    minimize    y^T Sigma y
    subject to  mu^T y == 1,  y >= 0
then w = y / sum(y). This is exactly equivalent to max-Sharpe in the
unconstrained case and a very good practical approximation under a
long-only constraint (the constraint that breaks exact equivalence is
y >= 0 vs w >= 0, which coincide here since sum(y) > 0). This is what
lets us solve "maximize Sharpe" as a QP instead of a nonlinear program.
"""

from __future__ import annotations
import numpy as np
import cvxpy as cp

ASSET_ORDER = ["equity", "bonds", "gold"]


def _psd_nudge(Sigma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Tiny ridge so CVXPY's solver never chokes on a near-singular
    sample covariance matrix (common with short lookback windows)."""
    return Sigma + eps * np.eye(Sigma.shape[0])


def max_sharpe_weights(mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0) -> np.ndarray:
    Sigma = _psd_nudge(Sigma)
    n = len(mu)
    excess = mu - rf
    y = cp.Variable(n, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, Sigma)), [excess @ y == 1])
    try:
        prob.solve(solver=cp.OSQP)
        if y.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(prob.status)
        w = np.maximum(y.value, 0)
        if w.sum() <= 1e-9:
            raise RuntimeError("degenerate solution")
        return w / w.sum()
    except Exception:
        # Fallback: if no asset has positive expected excess return (can
        # happen in a synthetic draw), max-Sharpe is ill-posed -> fall back
        # to min-variance instead of crashing the backtest.
        return min_variance_weights(Sigma)


def min_variance_weights(Sigma: np.ndarray, floor: dict[int, float] | None = None) -> np.ndarray:
    """Minimum-variance long-only portfolio, with optional per-asset floor
    constraints (used for the Crisis regime's mandatory gold allocation).
    floor: {asset_index: minimum_weight}
    """
    Sigma = _psd_nudge(Sigma)
    n = Sigma.shape[0]
    w = cp.Variable(n, nonneg=True)
    constraints = [cp.sum(w) == 1]
    if floor:
        for idx, min_w in floor.items():
            constraints.append(w[idx] >= min_w)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
    prob.solve(solver=cp.OSQP)
    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        # last-resort equal weight so the backtest never hard-crashes
        return np.ones(n) / n
    return np.clip(w.value, 0, None) / np.clip(w.value, 0, None).sum()


def weights_for_regime(regime: str, mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """The Dynamic Constraint Mapping entry point used by the backtester."""
    gold_idx = ASSET_ORDER.index("gold")
    if regime == "Bull":
        return max_sharpe_weights(mu, Sigma, rf=rf)
    elif regime == "Bear":
        return min_variance_weights(Sigma)
    elif regime == "Crisis":
        return min_variance_weights(Sigma, floor={gold_idx: 0.30})
    else:
        raise ValueError(f"Unknown regime: {regime}")
