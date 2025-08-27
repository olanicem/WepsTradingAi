#!/usr/bin/env python3
# ==========================================================
# 🧠 WEPSMasterDataLoader — Spiral Multi-Timeframe Edition
# ✅ Fetches 1h, 4h, 1d OHLCV from FMP
# ✅ Computes Sentiment Score per WEPS EPTS Framework
# ✅ Outputs unified multi-timeframe DataFrames per organism
# ✅ Robust error handling and logging for production
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import os
import requests
import pandas as pd
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("WEPS.DataLoader")
logger.setLevel(logging.INFO)

class WEPSMasterDataLoader:
    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("API key for Financial Modeling Prep is required.")

        self.sentiment_symbols = {
            "SPX": "SPY",
            "NIKKEI": "^N225",
            "VIX": "^VIX",
            "DXY": "USDX"
        }

    def fetch_ohlcv(self, symbol: str, timescale: str = "1day", limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch OHLCV data for given symbol and timescale.
        If limit is None or <=0, fetch all available data (subject to API constraints).
        """
        url = f"{self.BASE_URL}/historical-price-full/{symbol}?timescale={timescale}&apikey={self.api_key}"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "historical" not in data:
                raise ValueError(f"No historical data for {symbol} ({timescale})")

            df = pd.DataFrame(data["historical"])
            df = df[["date", "open", "high", "low", "close", "volume"]]
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values(by="date").reset_index(drop=True)

            if limit is not None and limit > 0:
                df = df.tail(limit).reset_index(drop=True)

            logger.debug(f"Fetched {len(df)} bars for {symbol} @ {timescale}")
            return df

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error fetching OHLCV for {symbol} at {url}: {http_err}")
            raise
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error fetching OHLCV for {symbol} at {url}: {req_err}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching OHLCV for {symbol}: {e}")
            raise

    def fetch_multi_timeframe_ohlcv(self, symbol: str, limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Fetch 1h, 4h, and 1d OHLCV for an organism."""
        timeframes = ["1hour", "4hour", "1day"]
        tf_map = {"1hour": "1h", "4hour": "4h", "1day": "1d"}

        data = {}
        for tf in timeframes:
            try:
                df = self.fetch_ohlcv(symbol, timescale=tf, limit=limit)
                data[tf_map[tf]] = df
            except Exception as e:
                logger.error(f"Failed to fetch {tf} OHLCV for {symbol}: {e}")
                data[tf_map[tf]] = pd.DataFrame()  # Empty DataFrame fallback

        return data

    def compute_daily_changes(self, df: pd.DataFrame) -> float:
        """Compute % change for most recent day."""
        if df.shape[0] < 2:
            return 0.0
        try:
            return (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2]
        except Exception as e:
            logger.warning(f"Error computing daily changes: {e}")
            return 0.0

    def fetch_sentiment_score(self) -> float:
        """Computes the WEPS EPTS sentiment score."""
        changes = {}
        for key, symbol in self.sentiment_symbols.items():
            try:
                df = self.fetch_ohlcv(symbol)
                changes[key] = self.compute_daily_changes(df)
            except Exception as e:
                logger.error(f"Failed to fetch OHLCV for sentiment symbol {symbol}: {e}")
                changes[key] = 0.0

        sentiment_score = (
            0.25 * changes.get("SPX", 0.0)
            + 0.35 * changes.get("NIKKEI", 0.0)
            - 0.3 * changes.get("VIX", 0.0)
            - 0.1 * changes.get("DXY", 0.0)
        )
        logger.info(f"Computed sentiment score: {sentiment_score:.6f}")
        return sentiment_score

    def fetch_organisms(self, organisms: List[str], limit: Optional[int] = None) -> Dict[str, Dict]:
        """Fetch multi-timeframe OHLCV + sentiment for selected organisms."""
        data = {}
        try:
            sentiment_score = self.fetch_sentiment_score()
        except Exception as e:
            logger.error(f"Failed to fetch sentiment score: {e}, defaulting to 0.0")
            sentiment_score = 0.0

        for symbol in organisms:
            try:
                multi_df = self.fetch_multi_timeframe_ohlcv(symbol, limit=limit)
                data[symbol] = {
                    "ohlcv_multi": multi_df,
                    "sentiment_score": sentiment_score
                }
                logger.info(f"Fetched data for organism {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch data for organism {symbol}: {e}")
                data[symbol] = {
                    "ohlcv_multi": {"1h": pd.DataFrame(), "4h": pd.DataFrame(), "1d": pd.DataFrame()},
                    "sentiment_score": sentiment_score
                }
        return data

# ==========================
# ⚡️ Usage Example
# ==========================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = WEPSMasterDataLoader()
    organisms = ["EURUSD", "USDJPY", "BTCUSD", "AAPL", "ETHUSD"]
    # Pass None or omit limit for full history
    data = loader.fetch_organisms(organisms, limit=None)

    for symbol, payload in data.items():
        print(f"\n⚡️ {symbol} - Sentiment Score: {payload['sentiment_score']:.4f}")
        for timeframe, df in payload["ohlcv_multi"].items():
            print(f"\n🕒 {symbol} [{timeframe}] Last 3 bars:\n{df.tail(3)}")
