"""
metrics.py — Performance Tear Sheet
=====================================
Standard risk-adjusted return metrics, computed the same way for the
strategy and both static benchmarks so the comparison is apples-to-apples.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(equity_curve: pd.Series) -> float:
    total_years = len(equity_curve) / TRADING_DAYS
    if total_years <= 0:
        return np.nan
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / total_years) - 1


def annualized_vol(equity_curve: pd.Series) -> float:
    rets = equity_curve.pct_change().dropna()
    return rets.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(equity_curve: pd.Series, rf: float = 0.0) -> float:
    rets = equity_curve.pct_change().dropna()
    excess = rets - rf / TRADING_DAYS
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)


def sortino_ratio(equity_curve: pd.Series, rf: float = 0.0) -> float:
    rets = equity_curve.pct_change().dropna()
    excess = rets - rf / TRADING_DAYS
    downside = excess[excess < 0]
    dd_std = downside.std()
    if dd_std == 0 or np.isnan(dd_std):
        return np.nan
    return (excess.mean() / dd_std) * np.sqrt(TRADING_DAYS)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


def calmar_ratio(equity_curve: pd.Series) -> float:
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return np.nan
    return annualized_return(equity_curve) / abs(mdd)


def tear_sheet_row(name: str, equity_curve: pd.Series, turnover_history: pd.Series | None = None) -> dict:
    row = {
        "Strategy": name,
        "Ann. Return": annualized_return(equity_curve),
        "Ann. Vol": annualized_vol(equity_curve),
        "Sharpe": sharpe_ratio(equity_curve),
        "Sortino": sortino_ratio(equity_curve),
        "Max Drawdown": max_drawdown(equity_curve),
        "Calmar": calmar_ratio(equity_curve),
    }
    if turnover_history is not None:
        n_years = len(equity_curve) / TRADING_DAYS
        row["Avg Annual Turnover"] = turnover_history.sum() / max(n_years, 1e-9)
    else:
        row["Avg Annual Turnover"] = np.nan
    return row


def build_tear_sheet(curves: dict[str, pd.Series], turnovers: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    turnovers = turnovers or {}
    rows = [tear_sheet_row(name, ec, turnovers.get(name)) for name, ec in curves.items()]
    tear = pd.DataFrame(rows).set_index("Strategy")
    pct_cols = ["Ann. Return", "Ann. Vol", "Max Drawdown", "Avg Annual Turnover"]
    for c in pct_cols:
        tear[c] = tear[c].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "n/a")
    for c in ["Sharpe", "Sortino", "Calmar"]:
        tear[c] = tear[c].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "n/a")
    return tear
