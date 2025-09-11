import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from prepare_data import TICKERS, BASELINE
from run import optimize_portfolio, calculate_returns, calculate_statistics


# Read baseline data (IBRX ETF)
baseline_data = pd.read_csv("baseline_data.csv", index_col=[0])
baseline_returns = (baseline_data[BASELINE] / baseline_data[BASELINE].shift(1))[1:]

# data with open prices for our tickers
open_data = baseline_data[TICKERS]

initial_notional = 100000
baseline_returns_acc = [initial_notional]

notional = initial_notional

for r in baseline_returns.values[1:]:
    notional *= r
    baseline_returns_acc.append(notional)

# create a dataframe with accumulated baseline prices, as if initial_notional was
# invested in the dataset's first day and held throughout the full timeframe
baseline_df = pd.DataFrame(
    index=baseline_returns.index, data=baseline_returns_acc, columns=["baseline"]
)
close_data = pd.read_csv("price_history.csv", index_col=[0])

notional_daily_values = []
day_i = 0
# now, for each day in the baseline history, we run our optimization,
# updating the weights every 63 days
for closing_date in baseline_df.index.values:
    window = close_data[TICKERS][close_data.index < closing_date]

    returns = calculate_returns(window)
    average_returns, cov_matrix = calculate_statistics(returns)

    w = np.random.random(len(TICKERS))
    w /= np.sum(w)

    if day_i == 0 or day_i == 63:
        if notional_daily_values == []:
            notional = initial_notional
        else:
            notional = notional_daily_values[-1]
        optimal = optimize_portfolio(w, returns, cov_matrix, to_optimize="sharpe")

        # distribute the notional N according to optimal distribution found
        notional_distribution = optimal * notional

        # buy shares at open price considering 0.5% of slippage
        total_shares_bought = notional_distribution / (
            open_data[TICKERS].loc[closing_date] * 1.005
        )
        day_i = 0
    day_i += 1

    # calculate the price of shares by the end of the day
    total_shares_value = total_shares_bought * close_data[TICKERS].loc[closing_date]

    # new notional is how much the portfolio is worth that day
    notional_daily_values.append(np.sum(total_shares_value))

baseline_df["Markowitz Model"] = notional_daily_values
baseline_df.plot(figsize=(20, 10))
plt.show()
