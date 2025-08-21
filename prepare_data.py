import yfinance as yf
import pandas as pd

TICKERS = ["PETR4.SA", "ITUB4.SA", "VALE3.SA", "SBSP3.SA", "ELET3.SA", "EMBR3.SA"]

BASELINE = "BOVA11.SA"
PRICE_HISTORY_FILENAME = "price_history.csv"
BASELINE_FILENAME = "baseline_data.csv"


def write_data():
    print("Downloading data for all tickers")
    raw_data = yf.download(
        " ".join(TICKERS), interval="1d", period="max", progress=False
    ).dropna()

    print("Saving downloaded data to csv")
    data = pd.DataFrame()
    for ticker in TICKERS:
        data[ticker] = raw_data["Close"][ticker]
    data.to_csv(PRICE_HISTORY_FILENAME)

    print("Downloading baseline data")
    raw_bova11 = yf.download(BASELINE, interval="1d", period="max", progress=False)
    data_bova11 = pd.DataFrame()
    data_bova11["BOVA11.SA"] = raw_bova11["Close"]["BOVA11.SA"]
    data_bova11.to_csv(BASELINE_FILENAME)


if __name__ == "__main__":
    write_data()
