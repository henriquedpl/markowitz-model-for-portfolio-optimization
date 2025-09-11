# Markowitz Modern Portfolio Theory (MPT) Implementation

This repository contains Python code implementing **Modern Portfolio Theory (MPT)** ([original paper](https://www.math.hkust.edu.hk/~maykwok/courses/ma362/07F/markowitz_JF.pdf)) developed by Harry Markowitz, applied to the Brazilian stock exchange. 
MPT is a quantitative framework for assembling a portfolio of assets such that the expected return is maximized for a given level of risk, defined as portfolio variance, or the risk is minimized, for a given level of return.

## Requirements

Main requirements are `matplotlib`, `numpy`, `scipy` and `yfinance`. You can install them by running `python install -r requirements.txt`


## How to run it
First, run `python prepare_data.py` to download the financial data for the stocks I picked for this experiment.
I picked a handful of companies from Brazil that met two requirements: first, they should be part of the BOVESPA index. 
Second, they should be well-stablished companies with a history that goes as far back as 2010. Also, I made sure to not 
pick two companies from a same market segment.

After running the script that downloads the data, you can run `python run.py` to generate some portfolio 
distributions and show them in a graph that crosses risk and return. They will be represented as circles, and there 
will also be a star for the optimized portfolio (the optimized portfolio has the highest sharpe ratio).

You can also run `python backtest.py` to run the experiment across all history of the IBRX ETF, comparing the 
optimized portfolio against buying and holding the ETF. It will then display a graph comparing the performance of both.
