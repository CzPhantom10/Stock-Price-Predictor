import pandas as pd
import numpy as np

def add_indicators(df):
    df["return"] = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(5).std()

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["target"] = df["close"].shift(-1)
    return df.dropna()
