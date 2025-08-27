# =============================================
# WEPS Spiral-Aware FMP Fetcher
# File: weps/utils/fmp_fetcher.py
# Author: Ola | WEPS Creator
# Description: Retrieves EOD OHLCV data for all
#              WEPS organisms across all asset classes
#              using confirmed FMP endpoint format.
# =============================================

import requests
import pandas as pd
import time
from datetime import datetime
from weps.utils.log_utils import log_event

FMP_BASE_URL = "https://financialmodelingprep.com/stable/historical-price-eod/full"
REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


def fetch_ohlcv_data(symbol: str, api_key: str, start_date: str, end_date: str, retry_limit: int = 3, retry_delay: int = 2) -> pd.DataFrame:
    """
    Fetches historical OHLCV data for any WEPS organism using final FMP endpoint.
    Applies spiral-grade validation and retry logic.
    """
    params = {
        "symbol": symbol,
        "from": start_date,
        "to": end_date,
        "apikey": api_key
    }

    attempt = 1
    while attempt <= retry_limit:
        try:
            response = requests.get(FMP_BASE_URL, params=params, timeout=10)
            if response.status_code != 200:
                raise ValueError(f"{symbol}: HTTP {response.status_code}")

            data = response.json()
            if not isinstance(data, list):
                raise ValueError(f"{symbol}: Unexpected response format.")

            df = pd.DataFrame(data)
            missing = REQUIRED_COLUMNS - set(df.columns)
            if missing:
                raise ValueError(f"{symbol}: Missing columns: {missing}")

            df = df[list(REQUIRED_COLUMNS)]
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            log_event(f"[DATA_FETCH] • {symbol:<8} | ✅ Success on attempt {attempt} | {len(df)} rows")
            return df

        except Exception as e:
            log_event(f"[DATA_FETCH] • {symbol:<8} | {symbol} fetch attempt {attempt} failed: {str(e)}")
            attempt += 1
            time.sleep(retry_delay)

    log_event(f"[SYSTEM    ] • GLOBAL   | [{symbol}] ❌ Critical error: {symbol} FMP fetch failed after {retry_limit} attempts.")
    return None
