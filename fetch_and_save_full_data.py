#!/usr/bin/env python3
"""
fetch_and_save_full_data.py
Institutional-grade script to fetch unlimited historical 1h, 4h, and 1d OHLCV data
for specified organisms using WEPSMasterDataLoader, saving CSVs for downstream use.

Author: Ola Bode (WEPS Creator)
"""

import os
import logging
from weps.data_loader import WEPSMasterDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WEPS.DataFetch")

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory {path}")

def fetch_and_save(organisms, base_path="~/weps/data/raw_ohlcv"):
    base_path = os.path.expanduser(base_path)
    ensure_dir(base_path)

    loader = WEPSMasterDataLoader()
    for org in organisms:
        logger.info(f"Starting fetch for {org}")
        try:
            # Fetch unlimited history by passing limit=None
            data = loader.fetch_multi_timeframe_ohlcv(org, limit=None)
            for tf, df in data.items():
                if df.empty:
                    logger.warning(f"No data fetched for {org} timeframe {tf}")
                    continue
                filename = os.path.join(base_path, f"{org.lower()}_{tf}_ohlcv.csv")
                df.to_csv(filename, index=False)
                logger.info(f"Saved {org} {tf} data to {filename} ({len(df)} rows)")
        except Exception as e:
            logger.error(f"Failed fetching or saving data for {org}: {e}", exc_info=True)

if __name__ == "__main__":
    # Add your full organism list here
    organisms = [
        "AAPL", "MSFT", "TSLA", "META", "GOOGL",
        "EURUSD", "GBPUSD", "USDJPY", "EURJPY", "AUDUSD",
        "BTCUSD", "ETHUSD", "DOGEUSD", "ADAUSD"
    ]
    fetch_and_save(organisms)
