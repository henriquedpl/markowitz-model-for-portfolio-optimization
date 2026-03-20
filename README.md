# Markowitz Model for Portfolio Optimization

This repository implements a practical **Modern Portfolio Theory (MPT)** workflow inspired by Harry Markowitz's framework, applied to a basket of Brazilian assets from Yahoo Finance data.

The project includes three main capabilities:

1. **Data ingestion** for historical market prices.
2. **Portfolio simulation + optimization** with Sharpe-focused weights.
3. **Historical backtest** comparing the optimized strategy vs. a benchmark ETF.

---

## What this project does

### 1) Download and persist market data (`prepare_data.py`)

The `prepare_data.py` script is responsible for collecting all input data used by the other scripts.

- Defines the universe of tradable assets in `TICKERS`:
  - `PETR4.SA`, `ITUB4.SA`, `VALE3.SA`, `SBSP3.SA`, `GGBR4.SA`, `B3SA3.SA`, `RADL3.SA`, `EMBJ`
- Defines the benchmark ETF in `BASELINE`:
  - `BRAX11.SA`
- Downloads full-period daily OHLCV data with `yfinance` (`period="max"`, `interval="1d"`).
- Writes two CSVs:
  - `price_history.csv`: daily **close** prices for all tickers (used by optimizer/simulation).
  - `baseline_data.csv`: benchmark close prices + ticker **open** prices (used by backtest to model entry fills).

Command:

```bash
python prepare_data.py
```

---

### 2) Build return/risk statistics and optimize portfolios (`run.py`)

The `run.py` script contains reusable portfolio math and a runnable experiment.

#### Data + return functions

- `read_data(filename="price_history.csv")`
  - Loads CSV data and raises a clear error if the file does not exist.
- `calculate_returns(price_history)`
  - Converts price history into **log returns** per ticker.
- `calculate_statistics(returns_data, ws=252)`
  - Computes annualized mean returns and covariance matrix over the last `WINDOW_SIZE` trading days (default: 252).

#### Portfolio valuation functions

- `w_expected_return_and_risk(...)`
  - For a given weight vector, computes:
    - expected annualized return,
    - volatility (standard deviation),
    - Sharpe proxy (`return / volatility`, no explicit risk-free rate).

#### Random portfolio generation

- `generate_portfolios(...)`
  - Samples `NUM_PORTFOLIOS` random long-only portfolios (default: 50,000),
  - Normalizes each weight vector to sum to 1,
  - Returns arrays of weights, expected returns, and risks.

#### Visualization

- `show_portfolios(...)`
  - Plots expected return vs. volatility scatter,
  - Colors points by Sharpe ratio,
  - Optionally overlays an optimized portfolio with a star marker.

#### Optimization engine (SciPy SLSQP)

- Objective helper functions:
  - `max_function_sharpe`: maximize Sharpe by minimizing negative Sharpe.
  - `max_function_return`: maximize expected return.
  - `min_function_risk`: minimize volatility.
- Constraint helpers:
  - Sum of weights must equal 1.
  - Optional risk/return constraints via neighborhood checks:
    - `MAX_ACCEPTED_RISK = 0.35`
    - `MIN_ACCEPTED_RETURN = 0.145`
- `optimize_portfolio(w0, returns, cov_matrix, to_optimize=...)`
  - Supports optimization modes:
    - `"sharpe"` (returns optimized weights),
    - `"expected_return"` (risk-constrained),
    - `"risk"` (return-constrained).

#### Script entrypoint behavior

Running `python run.py` will:

1. Read `price_history.csv`.
2. Compute returns/covariance.
3. Generate random portfolios.
4. Optimize one portfolio for Sharpe.
5. Display the risk-return chart.

Command:

```bash
python run.py
```

---

### 3) Historical strategy comparison (`backtest.py`)

The `backtest.py` script compares a Markowitz allocation strategy against buy-and-hold of `BRAX11.SA`.

#### Backtest logic summary

- Starts from `initial_notional = 100000`.
- Builds benchmark equity curve by compounding ETF daily return multipliers.
- For each date in benchmark history:
  1. Uses only historical closes prior to that date (`window`) to avoid look-ahead.
  2. Recomputes return/covariance statistics.
  3. Re-optimizes weights every 63 trading days (roughly quarterly).
  4. Applies `SLIPPAGE_RATE = 0.003` to open-price purchases.
  5. Marks holdings to close prices to get daily portfolio value.
- Produces a chart with both:
  - `baseline` (ETF buy-and-hold),
  - `Markowitz Model` (rebalanced optimized portfolio).

Command:

```bash
python backtest.py
```

---

## Project structure

```text
.
├── prepare_data.py   # Data download and CSV export
├── run.py            # Portfolio metrics, optimization, and scatter plot
├── backtest.py       # Rolling optimization backtest vs benchmark
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Installation

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

Main libraries used:

- `numpy`, `pandas` for data handling and matrix math.
- `scipy` for constrained optimization (`SLSQP`).
- `matplotlib` for visualizations.
- `yfinance` for market data retrieval.

---

## Typical end-to-end workflow

```bash
python prepare_data.py
python run.py
python backtest.py
```

1. Download raw market data.
2. Explore efficient-risk/return cloud + optimized point.
3. Evaluate strategy behavior through time against benchmark.

---

## Notes and assumptions

- Long-only portfolios with per-asset bounds `[0, 1]`.
- Portfolio weights are always constrained to sum to 1.
- Sharpe is approximated as `return / volatility` (risk-free rate not subtracted).
- Annualization uses a 252-trading-day convention.
- Backtest rebalances every 63 trading days and includes simple slippage modeling.
