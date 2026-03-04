import yfinance as yf
import pandas as pd

def download_data(ticker="AAPL", start="2015-01-01"):
    
    data = yf.download(ticker, start=start)

    data = data[["Open","High","Low","Close","Volume"]]

    data.to_csv("data/price_data.csv")

    return data


if __name__ == "__main__":
    df = download_data()
    print(df.head())
