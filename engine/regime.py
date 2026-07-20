"""
regime.py — Hidden Markov Regime Classifier
=============================================
Detects latent market regimes (Bull / Bear / Crisis) from observable features
using a Gaussian HMM, WITHOUT any manual labelling of history.

Feature vector (per trading day), all computed from data available AS OF
that day only:
    1. 21-day rolling equity return (momentum/trend proxy)
    2. 21-day annualized realized volatility of equity returns
    3. VIX level (z-scored against trailing 252-day window)

Why these three? They are cheap, liquid, always-available signals that
between them capture the two things that actually separate calm markets
from crises: *direction* (return) and *fear* (vol + VIX). Adding raw price
level would violate stationarity assumptions the HMM's Gaussian emissions
rely on, so we always feed it returns/vol/vol-of-vol style features, never
levels.

Label mapping: hmmlearn assigns arbitrary integer state IDs (0,1,2) — there
is no guarantee state 0 = Bull. We resolve this AFTER fitting by ranking the
fitted per-state mean return (highest -> Bull, lowest / most negative with
highest vol -> Crisis, remainder -> Bear).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

REGIME_NAMES = ["Bull", "Bear", "Crisis"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw prices/VIX into the stationary feature set the HMM trains on."""
    feat = pd.DataFrame(index=df.index)
    eq_ret = df["equity"].pct_change()

    feat["ret_21d"] = eq_ret.rolling(21).mean() * 21          # trend
    feat["vol_21d"] = eq_ret.rolling(21).std() * np.sqrt(252)  # realized annualized vol
    vix_mean = df["vix"].rolling(252, min_periods=42).mean()
    vix_std = df["vix"].rolling(252, min_periods=42).std()
    feat["vix_z"] = (df["vix"] - vix_mean) / vix_std

    return feat.dropna()


def _label_states(model: GaussianHMM, feature_cols: list[str]) -> dict[int, str]:
    """Rank fitted states by (mean return - mean vol - mean vix_z) to assign
    human-readable regime names. Highest score -> Bull, lowest -> Crisis.
    """
    means = model.means_  # shape (n_states, n_features)
    ret_idx = feature_cols.index("ret_21d")
    vol_idx = feature_cols.index("vol_21d")
    vixz_idx = feature_cols.index("vix_z")

    score = means[:, ret_idx] - means[:, vol_idx] - means[:, vixz_idx]
    order = np.argsort(score)[::-1]  # best (Bull) first
    mapping = {}
    for rank, state_id in enumerate(order):
        mapping[int(state_id)] = REGIME_NAMES[rank]
    return mapping


def fit_hmm(feature_window: pd.DataFrame, n_states: int = 3, seed: int = 42) -> tuple[GaussianHMM, dict[int, str]]:
    """Fit a fresh Gaussian HMM on a window of features (PAST DATA ONLY —
    caller is responsible for making sure no future information leaks in;
    see backtest.py's walk-forward loop for how that's enforced).
    """
    X = feature_window.to_numpy()
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=500,
        random_state=seed,
        tol=1e-4,
    )
    model.fit(X)
    label_map = _label_states(model, list(feature_window.columns))
    return model, label_map


def classify(model: GaussianHMM, label_map: dict[int, str], feature_window: pd.DataFrame) -> pd.Series:
    """Run the (already-fitted) Viterbi decoder over a feature window and
    return a Series of human-readable regime labels aligned to the index.
    """
    X = feature_window.to_numpy()
    state_seq = model.predict(X)  # Viterbi algorithm
    labels = pd.Series([label_map[s] for s in state_seq], index=feature_window.index, name="regime")
    return labels


def current_regime(model: GaussianHMM, label_map: dict[int, str], feature_window: pd.DataFrame) -> str:
    """Regime label for the LAST row only — this is the one we act on when
    rebalancing (using only information up to and including 'today').
    """
    return classify(model, label_map, feature_window).iloc[-1]
