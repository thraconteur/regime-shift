"""
data.py — Data Ingestion Layer
================================
Pulls the raw multi-asset universe + macro proxies needed by the rest of the
pipeline:

    - Equities   : SPY   (S&P 500 ETF)
    - Bonds      : AGG   (US Aggregate Bond ETF)   -> ballast / fixed income
    - Safe Haven : GLD   (Gold ETF)                -> crisis hedge
    - Vol proxy  : ^VIX  (CBOE Volatility Index)    -> regime feature
    - Macro      : FRED T10Y2Y (10y-2y yield spread), FRED BAMLH0A0HYM2
                   (high-yield credit spread)        -> regime feature

Design note for reviewers
--------------------------
This module hits Yahoo Finance (via `yfinance`) and FRED (via
`pandas_datareader`), both of which need outbound internet access. If those
calls fail (no internet, rate limit, symbol change, etc.) we fall back to a
synthetic-but-realistic regime-switching dataset so that the *rest of the
pipeline* (HMM fit, optimizer, walk-forward backtest) can still be developed,
unit-tested and demoed offline. The fallback is loud about the fact that it's
synthetic — it is never silently substituted for real data.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import warnings

ASSET_TICKERS = {
    "equity": "SPY",
    "bonds": "AGG",
    "gold": "GLD",
}
VIX_TICKER = "^VIX"
FRED_SERIES = {
    "yield_spread": "T10Y2Y",       # 10y-2y treasury spread (inversion -> recession risk)
    "credit_spread": "BAMLH0A0HYM2",  # ICE BofA US High Yield OAS (credit stress)
}


def fetch_live(start: str, end: str) -> pd.DataFrame:
    """Pull real daily adjusted-close prices + macro series and return a single
    wide DataFrame indexed by date. Raises if any leg fails — caller decides
    whether to fall back.
    """
    import yfinance as yf

    tickers = list(ASSET_TICKERS.values()) + [VIX_TICKER]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no data (no internet access from this environment?)")

    px = raw["Close"].copy()
    px = px.rename(columns={v: k for k, v in ASSET_TICKERS.items()})
    px = px.rename(columns={VIX_TICKER: "vix"})

    macro = pd.DataFrame(index=px.index)
    try:
        import pandas_datareader.data as web
        for name, series_id in FRED_SERIES.items():
            s = web.DataReader(series_id, "fred", start, end)
            macro[name] = s.reindex(px.index).ffill()
    except Exception as e:  # pragma: no cover - macro is a nice-to-have, not critical
        warnings.warn(f"FRED macro pull failed ({e}); continuing with price+VIX features only.")
        for name in FRED_SERIES:
            macro[name] = np.nan

    df = px.join(macro)
    df = df.ffill().dropna(subset=list(ASSET_TICKERS.keys()) + ["vix"])
    return df


def fetch_synthetic(start: str, end: str, seed: int = 42) -> pd.DataFrame:
    """Generate a SYNTHETIC but regime-realistic dataset for offline dev/testing.

    Simulates three latent regimes (Bull / Bear / Crisis) with a Markov
    transition matrix, then draws asset returns from regime-specific
    Gaussian parameters that mimic real stylized facts:
      - Bull:   positive equity drift, low vol, bonds/gold roughly flat
      - Bear:   negative equity drift, moderate vol, bonds rally slightly
      - Crisis: sharply negative equity drift, high vol, flight-to-quality
                into bonds and gold (negative equity-bond/gold correlation)

    THIS IS NOT REAL MARKET DATA. It exists purely so the modeling pipeline
    can be exercised end-to-end without network access. Swap in `fetch_live`
    for the real deliverable.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    # Regime transition matrix (rows sum to 1): Bull, Bear, Crisis
    P = np.array([
        [0.985, 0.013, 0.002],
        [0.030, 0.950, 0.020],
        [0.020, 0.080, 0.900],
    ])
    regime_params = {
        0: dict(eq=0.00045, eq_vol=0.007, bd=0.00006, bd_vol=0.0025, gd=0.0001, gd_vol=0.006, rho_eb=-0.05, rho_eg=0.0),
        1: dict(eq=-0.0006, eq_vol=0.013, bd=0.0002, bd_vol=0.003, gd=0.0002, gd_vol=0.008, rho_eb=-0.2, rho_eg=0.1),
        2: dict(eq=-0.0022, eq_vol=0.028, bd=0.0004, bd_vol=0.004, gd=0.0006, gd_vol=0.012, rho_eb=-0.45, rho_eg=0.35),
    }

    states = np.zeros(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(3, p=P[states[t - 1]])

    eq_ret = np.zeros(n)
    bd_ret = np.zeros(n)
    gd_ret = np.zeros(n)
    for t in range(n):
        p = regime_params[states[t]]
        cov = np.array([
            [p["eq_vol"] ** 2, p["rho_eb"] * p["eq_vol"] * p["bd_vol"], p["rho_eg"] * p["eq_vol"] * p["gd_vol"]],
            [p["rho_eb"] * p["eq_vol"] * p["bd_vol"], p["bd_vol"] ** 2, 0.05 * p["bd_vol"] * p["gd_vol"]],
            [p["rho_eg"] * p["eq_vol"] * p["gd_vol"], 0.05 * p["bd_vol"] * p["gd_vol"], p["gd_vol"] ** 2],
        ])
        mean = np.array([p["eq"], p["bd"], p["gd"]])
        draw = rng.multivariate_normal(mean, cov)
        eq_ret[t], bd_ret[t], gd_ret[t] = draw

    equity = 100 * np.cumprod(1 + eq_ret)
    bonds = 100 * np.cumprod(1 + bd_ret)
    gold = 100 * np.cumprod(1 + gd_ret)

    # VIX proxy: rescaled trailing realized vol of the synthetic equity leg
    # (np.nanmax, not .max() — a plain ndarray .max() propagates the single
    # NaN from the first rolling-window observation and silently NaNs out
    # the whole series)
    roll_vol = pd.Series(eq_ret).rolling(10, min_periods=1).std().bfill().to_numpy()
    vix = 10 + 100 * roll_vol / np.nanmax(roll_vol) * 3.0
    vix = np.clip(vix, 9, 85)

    df = pd.DataFrame(
        {
            "equity": equity,
            "bonds": bonds,
            "gold": gold,
            "vix": vix,
            "yield_spread": 1.0 - 3.0 * (states == 2).astype(float) * rng.uniform(0.3, 1.0, n),
            "credit_spread": 3.0 + 12.0 * (states >= 1).astype(float) * rng.uniform(0.4, 1.0, n),
            "_true_regime": states,  # kept ONLY for synthetic sanity-checks, not used by the model
        },
        index=dates,
    )
    return df


def load_dataset(start: str, end: str, use_synthetic_fallback: bool = True) -> tuple[pd.DataFrame, bool]:
    """Try live data first; fall back to synthetic if requested and live fails.
    Returns (dataframe, is_synthetic_flag).
    """
    try:
        return fetch_live(start, end), False
    except Exception as e:
        if not use_synthetic_fallback:
            raise
        warnings.warn(
            f"Live data fetch failed ({e}). Falling back to SYNTHETIC data for pipeline "
            "development/testing. Re-run with internet access (e.g. Google Colab or your "
            "local machine) to get the real result."
        )
        return fetch_synthetic(start, end), True
