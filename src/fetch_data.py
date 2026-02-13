import os
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()

# Alpaca API keys
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("Alpaca API keys not found. Check your .env file.")

# Stock and date range
SYMBOLS = ["NVDA", "AAPL", "TSLA"]
START_DATE = "2023-01-01"
END_DATE = "2026-02-02"

# Create Alpaca client
client = StockHistoricalDataClient(API_KEY, API_SECRET)

for symbol in SYMBOLS:
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=START_DATE,
        end=END_DATE,
    )

    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()
    df = df[df["symbol"] == symbol]
    df = df.sort_values("timestamp")

    df.to_csv(f"../data/raw_data_{symbol}.csv", index=False)
    print(f"{symbol} data saved")
