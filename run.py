import os
import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt


from prepare_data import TICKERS


WINDOW_SIZE = 252
NUM_PORTFOLIOS = 50000


def read_data(filename="price_history.csv"):
    """
    Read the csv file containing the price history for all tickers.
    Looks for a file called <filename>, complaining if it does not exist.

    Args:
        filename: name of the file, with extension
    Returns:
        a pandas dataframe with the data
    """
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=[0])
    else:
        raise FileNotFoundError(
            f"File {filename} does not exist. Try running prepare_data.py first."
        )


def calculate_returns(price_history):
    """
    Calculate the returns based on price history, using log returns.

    Args:
        price_history: a dataframe with price history for all tickers
    Returns:
        a dataframe with size n-1 with log returns for all tickers
    """
    if price_history is None:
        price_history = read_data()

    returns_data = pd.DataFrame()
    for ticker in price_history.columns:
        returns_data[ticker] = np.log(
            price_history[ticker] / price_history[ticker].shift(1)
        )[1:]
    return returns_data


def calculate_statistics(returns_data):
    """
    Based on the returns calculated in the function above, calculates the
    average returns (simple mean across all returns, grouped by ticker) and a
    covariance matrix. Since I picked stocks with little correlation (e.g. from
    different segments the covariance between them will be pretty low -- |cov(x,y)| < 0.2)

    Params:
        returns_data: a dataframe with daily returns
    Returns:
        an array with average daily returns and a covariance matrix
    """
    average_returns = returns.mean() * WINDOW_SIZE
    cov_matrix = returns.cov() * WINDOW_SIZE
    return average_returns, cov_matrix


def w_expected_return_and_risk(returns, cov_matrix, portfolio_weights, print_out=False):
    """
    Given a certain set of weights for each ticker representing a portfolio,
    calculate the expected return and the associated risk (volatility) of that
    portfolio.
    Params:
        returns: daily returns dataframe
        cov_matrix: covariance matrix
        portfolio_weights: an 1d array representing weights (whose items should
            add up to 1)
    """
    # annual returns
    portfolio_return = np.sum(returns.mean() * portfolio_weights) * WINDOW_SIZE
    portfolio_volatility = np.sqrt(
        np.dot(portfolio_weights.T, np.dot(cov_matrix, portfolio_weights))
    )
    if print_out:
        print(f"Expected portfolio mean (return): {portfolio_return}")
        print(f"Expected portfolio volatility (stddev): {portfolio_volatility}")

    return (
        portfolio_return,
        portfolio_volatility,
        portfolio_return / portfolio_volatility,  # sharpe ratio
    )


def generate_portfolios(returns, cov_matrix):
    """
    Generates a series of portfolios with random weights, and returns
    the weights, the expected returns and the associated risk for each portfolio

    Params:
        returns: daily returns dataframe
        cov_matrix: covariance matrix
    Returns:
        portfolio_weights: weights of each portfolio
        portfolio_means: expected returns of each portfolio
        portfolio_risks: volatility of each portfolio
    """
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(len(TICKERS))
        w /= np.sum(w)
        portfolio_weights.append(w)
        w_return, w_risk, _ = w_expected_return_and_risk(returns, cov_matrix, w)
        portfolio_means.append(w_return)
        portfolio_risks.append(w_risk)

    return (
        np.array(portfolio_weights),
        np.array(portfolio_means),
        np.array(portfolio_risks),
    )


def show_portfolios(w_returns, w_volatilities, returns, cov_matrix, optimal=None):
    """
    Show portfolios using matplotlib
    Also plots a star with the optimal portfolio (with higher sharpe ratio),
    if one is passed as a parameter

    Params:
        w_returns: expected returns for each generated portfolio
        w_volatilities: array of volatilities for each generated portfolio
        returns: daily returns df
        cov_matrix: covariance matrix
        optimal: optimal weights found by scipy's optimization method
    """

    plt.figure(figsize=(10, 6))
    plt.scatter(w_volatilities, w_returns, c=w_returns / w_volatilities, marker="o")
    plt.grid(True)
    plt.xlabel("Expected volatility")
    plt.ylabel("Expected return")
    plt.colorbar(label="Sharpe ratio")

    if optimal is not None:
        optimal_return, optimal_risk, optimal_sharpe = w_expected_return_and_risk(
            returns, cov_matrix, optimal
        )
        plt.plot(optimal_risk, optimal_return, "g*", markersize=15)
    plt.show()


def max_function_sharpe(w0, args):
    """
    Since scipy can not maximize a function, we will minimize its negative.
    This should be used exclusively by optimize_portfolio().
    """
    returns, cov_matrix = args
    _, _, sharpe_ratio = w_expected_return_and_risk(returns, cov_matrix, w0)
    return -sharpe_ratio


def optimize_portfolio(w0, returns, cov_matrix, to_optimize="sharpe"):
    """
    Find the optimal portfolio, i.e. the one with highest sharpe ratio.

    Params:
        w0: initial set of portfolio weights
        cov_matrix: covariance matrix
        to_optimize: what to optimize the portfolio for. in the future I'll add
         also "returns" and "risk"

    Returns:
        a set of weights w for the optimal portfolio
    """
    # make sure the sum of the weights is 1
    w_sum_constraint = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    # risk_constraint = {"type", "ineq", "fun": lambda x: w_expected_return_and_risk}
    bounds = tuple((0, 1) for _ in range(len(TICKERS)))
    if to_optimize == "sharpe":
        return optimize.minimize(
            fun=max_function_sharpe,
            x0=w0,
            args=[returns, cov_matrix],
            method="SLSQP",
            bounds=bounds,
            constraints=w_sum_constraint,
        )["x"]
    elif to_optimize == "expected_return":
        pass
    elif to_optimize == "risk":
        pass


# main loop
price_history = read_data()
returns = calculate_returns(price_history)
average_returns, cov_matrix = calculate_statistics(returns)

weights, means, risks = generate_portfolios(returns)
optimal = optimize_portfolio(weights[0], returns, cov_matrix)
# print(optimal)
show_portfolios(means, risks, returns, cov_matrix, optimal)
