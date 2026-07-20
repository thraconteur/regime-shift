"""
backtest.py — Walk-Forward Validation Harness
================================================
This is the module that stops the project from "hallucinating alpha".
Two specific traps it exists to avoid:

1. LOOK-AHEAD BIAS
   At every rebalance date t, the HMM is RE-FIT from scratch using only
   rows up to and including t (see `_refit_hmm_up_to`). We never fit once
   on the full history and then walk backwards through it — that would let
   the model "know" about a 2020-style crash while classifying 2019 data.
   Expected returns (mu) and covariance (Sigma) fed to the optimizer are
   likewise estimated only from a trailing lookback window ending at t.

2. PORTFOLIO THRASHING / TRANSACTION COSTS
   Every time weights change at a rebalance, turnover = sum(|w_new - w_drifted|)
   is computed and a cost (in bps) is deducted directly from that period's
   portfolio return. Without this, an HMM that flickers between regimes
   would look great on paper and lose money in reality on trading costs
   alone — modelling this explicitly is the whole point of Goal #4 in the
   brief.

Regime-conditional mu/Sigma estimation
----------------------------------------
Instead of just using the flat trailing-window mean/covariance, we ALSO
compute mean/covariance using only the historical days (within the trailing
window) that the HMM itself classified as being in the SAME regime as
"today". This is a genuinely more defensible estimate — a Crisis day's
optimizer shouldn't be diluted by mixing in Bull-day statistics — and falls
back to the flat trailing-window estimate if there aren't enough
regime-matched samples yet (min_regime_days).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from . import regime as regime_mod
from . import optimizer as opt_mod

ASSET_COLS = ["equity", "bonds", "gold"]


def _refit_hmm_up_to(df: pd.DataFrame, as_of_idx: int, min_train_days: int):
    """Build features and fit a brand-new HMM using ONLY rows [0, as_of_idx]."""
    hist = df.iloc[: as_of_idx + 1]
    feats = regime_mod.build_features(hist)
    if len(feats) < min_train_days:
        return None, None, None
    model, label_map = regime_mod.fit_hmm(feats)
    labels = regime_mod.classify(model, label_map, feats)
    return model, label_map, labels


def _estimate_mu_sigma(returns: pd.DataFrame, labels: pd.Series, today_regime: str,
                        lookback: int, min_regime_days: int, ann_factor: int = 252):
    """Trailing-window mu/Sigma, refined with a regime-conditional overlay
    when enough same-regime observations exist."""
    trailing = returns.tail(lookback)
    mu_flat = trailing.mean().to_numpy() * ann_factor
    Sigma_flat = trailing.cov().to_numpy() * ann_factor

    matched_idx = labels.tail(lookback)[labels.tail(lookback) == today_regime].index
    matched_idx = matched_idx.intersection(trailing.index)
    if len(matched_idx) >= min_regime_days:
        regime_slice = trailing.loc[matched_idx]
        mu = regime_slice.mean().to_numpy() * ann_factor
        Sigma = regime_slice.cov().to_numpy() * ann_factor
        # Guard against a near-singular covariance from too few points
        if np.all(np.isfinite(Sigma)) and np.linalg.det(Sigma + 1e-6 * np.eye(3)) > 1e-12:
            return mu, Sigma, True
    return mu_flat, Sigma_flat, False


def run_walk_forward_backtest(
    df: pd.DataFrame,
    rebalance_every: int = 21,       # trading days between rebalances (~monthly)
    initial_train_days: int = 504,   # ~2 years before the first rebalance
    lookback_days: int = 126,        # ~6 months for mu/Sigma estimation
    min_regime_days: int = 20,
    cost_bps: float = 7.5,           # transaction friction per unit turnover
    rf_annual: float = 0.0,
) -> dict:
    """Runs the full engine: refit HMM -> classify regime -> map to objective
    -> optimize -> apply weights with turnover cost -> step forward.

    Returns a dict with the equity curve, regime history, weight history and
    turnover history — everything `metrics.py` needs to build the tear sheet.
    """
    prices = df[ASSET_COLS]
    rets = prices.pct_change().dropna()

    dates = rets.index
    n = len(dates)

    equity_curve = pd.Series(index=dates, dtype=float)
    equity_curve.iloc[0] = 1.0
    weight_history = pd.DataFrame(index=dates, columns=ASSET_COLS, dtype=float)
    regime_history = pd.Series(index=dates, dtype=object)
    turnover_history = pd.Series(0.0, index=dates)

    current_w = np.array([1 / 3, 1 / 3, 1 / 3])  # start equal-weight before first rebalance
    last_rebalance_pos = -10**9

    for i, dt in enumerate(dates):
        # Absolute position in the *original* df (features need price history
        # before the first return too), so map i -> df index.
        as_of_idx = df.index.get_loc(dt)

        do_rebalance = (i - last_rebalance_pos >= rebalance_every) and (as_of_idx + 1 >= initial_train_days)

        if do_rebalance:
            model, label_map, labels = _refit_hmm_up_to(df, as_of_idx, min_train_days=initial_train_days // 2)
            if model is not None:
                today_regime = labels.iloc[-1]
                mu, Sigma, used_regime_cond = _estimate_mu_sigma(
                    rets.loc[:dt], labels, today_regime, lookback_days, min_regime_days
                )
                target_w = opt_mod.weights_for_regime(today_regime, mu, Sigma, rf=rf_annual)
                regime_history.loc[dt] = today_regime
            else:
                target_w = current_w
                regime_history.loc[dt] = "Warming up"

            turnover = np.abs(target_w - current_w).sum()
            cost = (cost_bps / 1e4) * turnover
            equity_curve.iloc[i] = (equity_curve.iloc[i - 1] if i > 0 else 1.0) * (1 - cost)
            current_w = target_w
            turnover_history.iloc[i] = turnover
            last_rebalance_pos = i
        else:
            regime_history.loc[dt] = regime_history.iloc[i - 1] if i > 0 else "Warming up"
            if i == 0:
                equity_curve.iloc[i] = 1.0

        # Apply today's asset returns to the (possibly just-updated) weights,
        # then let weights drift with relative performance until next rebalance.
        if i > 0 and not do_rebalance:
            day_ret = float(np.dot(current_w, rets.loc[dt].to_numpy()))
            equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 + day_ret)
            # drift weights
            asset_growth = 1 + rets.loc[dt].to_numpy()
            drifted = current_w * asset_growth
            current_w = drifted / drifted.sum()
        elif do_rebalance:
            day_ret = float(np.dot(current_w, rets.loc[dt].to_numpy()))
            equity_curve.iloc[i] = equity_curve.iloc[i] * (1 + day_ret)
            asset_growth = 1 + rets.loc[dt].to_numpy()
            drifted = current_w * asset_growth
            current_w = drifted / drifted.sum()

        weight_history.loc[dt] = current_w

    return {
        "equity_curve": equity_curve,
        "regime_history": regime_history,
        "weight_history": weight_history,
        "turnover_history": turnover_history,
        "returns": rets,
    }


def run_static_benchmark(df: pd.DataFrame, weights: dict[str, float], rebalance_every: int = 21) -> pd.Series:
    """Buy-and-hold-with-periodic-rebalance benchmark (e.g. 60/40, equal-weight).
    No transaction costs applied — this is the classic textbook baseline the
    strategy needs to beat NET of its own costs to be worth building at all.
    """
    prices = df[ASSET_COLS]
    rets = prices.pct_change().dropna()
    w0 = np.array([weights.get(a, 0.0) for a in ASSET_COLS])

    equity_curve = pd.Series(index=rets.index, dtype=float)
    equity_curve.iloc[0] = 1.0
    current_w = w0.copy()

    for i, dt in enumerate(rets.index):
        if i > 0 and i % rebalance_every == 0:
            current_w = w0.copy()
        day_ret = float(np.dot(current_w, rets.loc[dt].to_numpy()))
        equity_curve.iloc[i] = (equity_curve.iloc[i - 1] if i > 0 else 1.0) * (1 + day_ret)
        if i > 0:
            asset_growth = 1 + rets.loc[dt].to_numpy()
            drifted = current_w * asset_growth
            current_w = drifted / drifted.sum()

    return equity_curve
