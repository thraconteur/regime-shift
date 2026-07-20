# REGIME-SHIFT
### Macro-Aware Tactical Asset Allocation Engine

A dynamic 3-asset (Equities / Bonds / Gold) allocation engine that detects
hidden market regimes with a Hidden Markov Model and switches the portfolio
optimizer's objective function per regime, validated with a strict
walk-forward harness and explicit transaction-cost modelling.

---

## 1. What's actually in this repo

```
regime-shift/
├── engine/
│   ├── data.py         # data ingestion (yfinance + FRED, synthetic fallback)
│   ├── regime.py       # feature engineering + Gaussian HMM regime classifier
│   ├── optimizer.py    # CVXPY regime-conditional portfolio optimizer
│   ├── backtest.py     # walk-forward validation harness + benchmarks
│   └── metrics.py      # Sharpe / Sortino / Calmar / drawdown tear sheet
├── regime_shift.ipynb  # main pipeline notebook (data → regime → optimize → backtest → report)
└── README.md
```

`engine/` is a plain importable package rather than one giant notebook cell
soup — each module has one job, so you can unit test the HMM fitting logic
independently of the optimizer, or swap the optimizer for a different one,
without touching the backtest loop.

## 2. Setup

```bash
python -m venv venv && source venv/bin/activate   # or use conda
pip install -r requirements.txt
jupyter notebook regime_shift.ipynb
```

The notebook needs outbound internet access (Yahoo Finance + FRED). Run it in
Google Colab or any machine with a normal internet connection — a fully
offline sandbox will trigger the synthetic-data fallback described below
instead of pulling real prices.

## 3. Architecture decisions (the "why", not just the "what")

**Why a Gaussian HMM and not a rule-based regime filter (e.g. "Bear = SPY
down >20% from high")?** Rule-based thresholds are backward-looking by
construction — by the time a 20% drawdown is confirmed, most of the damage
is already done. An HMM instead learns a *joint* probability distribution
over returns, volatility and VIX-implied fear, and the Viterbi algorithm
gives the single most likely regime *path* given everything observed up to
today — it reacts to changing statistical structure, not a single
after-the-fact threshold.

**Why feed the HMM `return / vol / VIX-z` and never raw price levels?**
Gaussian HMM emissions assume each state's observations are drawn from a
roughly stationary Gaussian. Price levels trend upward over decades (never
stationary); returns, rolling vol, and a z-scored VIX are all
(approximately) stationary, which is what makes the state means/covariances
fitted by the HMM actually meaningful.

**Why refit the HMM from scratch at every rebalance instead of fitting once
on the full history?** Fitting once on the full sample and then "looking
back" at how each historical date was classified is look-ahead bias: the
model fit on 2010–2024 data implicitly knows about the 2020 COVID crash
while "classifying" January 2015. `engine/backtest.py`'s walk-forward loop
refits a brand-new HMM using only `data[:t]` at every single rebalance date,
which is computationally more expensive but is the only way the backtest's
numbers mean anything out-of-sample.

**Why the Cornuejols–Tütüncü reformulation for max-Sharpe instead of
scipy.optimize?** Sharpe ratio is quasi-concave, not concave, in portfolio
weights, so it isn't a natural fit for a standard convex solver. The
reformulation (`y = w / κ`, minimize `y'Σy` subject to `μ'y = 1, y ≥ 0`,
then renormalize) turns it into a QP that CVXPY's `OSQP` backend solves
reliably and fast — no nonlinear solver, no local-optima risk. Full
derivation is in the `optimizer.py` docstring.

**Why charge transaction costs as `turnover × bps` rather than ignoring
them?** A regime-switching model that flickers between states every few
days will look fantastic gross of costs and lose money net of them. Explicit
turnover costing is what turns this from a "can I fit an HMM" exercise into
a "would this survive contact with a real trading desk" exercise — which is
the whole point of the brief.

**Why regime-conditional `μ`/`Σ` estimation on top of a flat trailing
window?** A flat 6-month lookback mixes Crisis-day statistics in with
Bull-day statistics if the regime just changed. Filtering the lookback down
to only the days that were themselves classified in the *current* regime
gives the optimizer inputs that are more representative of "what a Crisis
day actually looks like" — with a safe fallback to the flat window when
there isn't enough regime-matched history yet.

## 4. Known limitations (see notebook Section 7 for the full discussion)

- 3-asset mean-variance optimization is prone to corner solutions when `μ`/`Σ`
  are noisily estimated; a production version would cap single-asset weight.
- The synthetic-data fallback (see below) exists purely for offline pipeline
  testing — it is not a real backtest and the notebook prints a loud warning
  whenever it's active.
- Gaussian HMM state labels can occasionally flip between adjacent refits on
  genuinely ambiguous data; this is inherent to unsupervised regime
  detection, not a implementation bug.

## 5. Testing / reproducibility

- `test_pipeline.py` runs the full pipeline against the synthetic dataset and
  prints a tear sheet — use it as a smoke test after any change to `engine/`.
- Every HMM fit is seeded (`random_state=42`) and the synthetic data
  generator is seeded too, so results are reproducible run-to-run given the
  same real market data snapshot.
- `regime_shift.ipynb` was executed top-to-bottom with `jupyter nbconvert
  --execute` before being committed, so the saved outputs are guaranteed to
  match what the code currently produces.

## 6. About the synthetic-data fallback

`engine/data.py`'s `load_dataset()` tries `yfinance` + FRED first. If that
fails for any reason (no internet, a symbol delisting, a rate limit), it
falls back to `fetch_synthetic()`, which simulates a 3-state Markov regime
process with realistic stylized facts (positive-drift/low-vol Bull,
negative-drift/moderate-vol Bear, sharply-negative/high-vol Crisis with
flight-to-quality correlation shifts into bonds and gold). This exists
*solely* so the modelling pipeline (HMM fit, optimizer, walk-forward harness)
can be built, tested and demoed without a live data connection. It is never
silently substituted — `IS_SYNTHETIC` is checked and printed at the top of
the notebook, and every chart title flags it.
