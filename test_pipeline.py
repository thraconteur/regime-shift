import warnings
import pandas as pd
from engine import data, backtest, metrics

warnings.filterwarnings("ignore")

df, is_synthetic = data.load_dataset("2012-01-01", "2024-01-01")
print("is_synthetic:", is_synthetic, "| rows:", len(df))
print(df.head())

results = backtest.run_walk_forward_backtest(df, cost_bps=7.5)
print("\nEquity curve tail:")
print(results["equity_curve"].tail())
print("\nRegime value counts:")
print(results["regime_history"].value_counts())

bench_6040 = backtest.run_static_benchmark(df, {"equity": 0.6, "bonds": 0.4})
bench_eq = backtest.run_static_benchmark(df, {"equity": 1/3, "bonds": 1/3, "gold": 1/3})

tear = metrics.build_tear_sheet(
    {
        "Regime-Shift": results["equity_curve"],
        "60/40": bench_6040,
        "Equal-Weight": bench_eq,
    },
    turnovers={"Regime-Shift": results["turnover_history"]},
)
print("\nTear sheet:")
print(tear)
