import yfinance as yf
import pandas as pd

tickers = ["PETR4.SA", "ITUB4.SA", "VALE3.SA", "SBSP3.SA", "ELET3.SA"]
data = {}
# download all data for all tickers

print(f"Downloading data for all tickers")
raw_data = yf.download(
    " ".join(tickers), interval="1d", period="max", auto_adjust=True, progress=False
).dropna()

print("Saving downloaded data to csv")
data = pd.DataFrame()
for ticker in tickers:
    data[ticker] = raw_data["Close"][ticker]
data.to_csv("complete_ticker_data.csv")

# to read: pd.read_csv('ticker_data.csv', index_col=[0])
