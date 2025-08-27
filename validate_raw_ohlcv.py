#!/usr/bin/env python3
import os
import pandas as pd

DATA_DIR = os.path.expanduser("~/weps/data/raw_ohlcv")
ASSETS = ["aapl", "msft", "tsla", "meta", "googl", "eurusd", "gbpusd", "usdjpy",
          "eurjpy", "audusd", "btcusd", "ethusd", "dogeusd", "adausd"]
TIMEFRAMES = ["1h", "4h", "1d"]

def validate_csv(asset: str, timeframe: str):
    filename = f"{asset}_{timeframe}_ohlcv.csv"
    filepath = os.path.join(DATA_DIR, filename)
    print(f"Validating {filepath} ...")

    if not os.path.isfile(filepath):
        print(f"  ERROR: File not found: {filepath}")
        return

    df = pd.read_csv(filepath)

    # Ensure date column exists
    if 'date' not in df.columns:
        print(f"  ERROR: Missing 'date' column in {filename}")
        return

    # Convert 'date' column to datetime if not index
    if df.index.name != 'date':
        try:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        except Exception as e:
            print(f"  ERROR: Failed to convert 'date' to datetime index: {e}")
            return

    # Check index dtype
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        print(f"  ERROR: Index is not datetime64 dtype in {filename}")
        return

    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"  WARNING: Found {missing} missing values in {filename}")

    print(f"  Rows: {len(df)}, Columns: {df.shape[1]}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print("  Sample data:")
    print(df.head(2))
    print()

if __name__ == "__main__":
    for asset in ASSETS:
        for tf in TIMEFRAMES:
            validate_csv(asset, tf)
