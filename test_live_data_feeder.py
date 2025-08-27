from weps.utils.live_data_feeder import LiveDataFeeder
import os
import pandas as pd

def test_live_data_feeder(organism="AAPL"):
    base_path = os.path.expanduser("~/weps/data/raw_ohlcv")
    filepath = os.path.join(base_path, f"{organism.lower()}_ohlcv.csv")
    df = pd.read_csv(filepath, parse_dates=["date"])
    
    hist_data = {"1d": df, "1h": df.copy(), "4h": df.copy()}  # replicate daily for hourly frames as placeholder
    
    feeder = LiveDataFeeder(organism, timeframes=["1h", "4h", "1d"])
    feeder.initialize(hist_data)
    
    print(f"Initialized LiveDataFeeder for {organism} with {len(df)} rows.")
    
    for i in range(5):
        state_info = feeder.step()
        print(f"Step {i+1}:")
        print(f"  State vector shape: {state_info.get('state_vector').shape}")
        print(f"  Done flag: {state_info.get('done')}")
        print(f"  Current price: {state_info.get('info', {}).get('price')}")

if __name__ == "__main__":
    test_live_data_feeder()
