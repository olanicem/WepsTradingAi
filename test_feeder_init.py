#!/usr/bin/env python3
import pandas as pd
from weps.utils.live_data_feeder import LiveDataFeeder

def test_feeder_init_with_csv(asset_symbol: str):
    base_path = "/home/llknogunsola/weps/data/raw_ohlcv"
    timeframes = ["1h", "4h", "1d"]
    data = {}

    for tf in timeframes:
        file_path = f"{base_path}/{asset_symbol.lower()}_{tf}_ohlcv.csv"
        df = pd.read_csv(file_path, parse_dates=["date"])
        df.set_index("date", inplace=True)
        print(f"{tf} DF index dtype for {asset_symbol}: {df.index.dtype}")
        data[tf] = df

    feeder = LiveDataFeeder(asset_symbol, timeframes)
    feeder.initialize(data)
    print(f"LiveDataFeeder initialized for {asset_symbol} with data lengths: {[len(data[tf]) for tf in timeframes]}")

if __name__ == "__main__":
    test_feeder_init_with_csv("AAPL")
