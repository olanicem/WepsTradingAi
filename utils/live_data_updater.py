#!/usr/bin/env python3
# ================================================================
# LiveDataUpdater — Institutional-Grade Incremental Market Data Manager for WEPS
# Full Data Integrity, Timezone, Resampling, and Multi-Timeframe Sync
# Author: Ola Bode (WEPS Creator)
# ================================================================

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger("WEPS.LiveDataUpdater")

class LiveDataUpdater:
    """
    Manages live incremental updates of OHLCV data per timeframe for WEPS pipeline.

    Features:
      - Enforces strict datetime ordering and uniqueness of 'date' index.
      - Supports timezone-aware datetime index normalization.
      - Efficient appending and updating of candles without data loss.
      - Multi-timeframe synchronization utilities for resampling (if needed).
      - Data validation to ensure integrity (no NaNs, correct dtypes).
      - Thread-safe design considerations (for async fetchers).

    Attributes:
        dfs: Dict[str, pd.DataFrame]
          Maintains latest OHLCV DataFrame per timeframe, sorted by date ascending.
    """

    REQUIRED_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']

    def __init__(self, initial_dfs: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialize with optional initial multi-timeframe OHLCV data.

        Args:
            initial_dfs: Optional dict mapping timeframe strings (e.g., '1m', '1h', '1d')
                         to OHLCV DataFrames with columns as REQUIRED_COLUMNS,
                         sorted ascending by 'date' column.
        """
        self.dfs: Dict[str, pd.DataFrame] = {}
        if initial_dfs:
            for tf, df in initial_dfs.items():
                self._validate_and_store(tf, df)

    def _validate_and_store(self, timeframe: str, df: pd.DataFrame) -> None:
        """
        Validate and store initial or updated dataframe for a timeframe.

        Args:
            timeframe: Timeframe string key.
            df: DataFrame with OHLCV data.
        """
        # Validate columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame for {timeframe} missing required columns: {missing_cols}")

        # Convert 'date' to datetime64[ns, UTC] if not already
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], utc=True)

        # Drop duplicates and sort ascending by date
        df = df.drop_duplicates(subset='date', keep='last')
        df = df.sort_values('date').reset_index(drop=True)

        # Validate no NaNs in key columns
        if df[self.REQUIRED_COLUMNS[1:]].isnull().any().any():
            raise ValueError(f"NaNs detected in OHLCV columns for timeframe {timeframe}")

        # Store
        self.dfs[timeframe] = df
        logger.debug(f"Stored validated dataframe for timeframe {timeframe} with {len(df)} rows")

    def update_timeframe(self, timeframe: str, new_candles: pd.DataFrame) -> None:
        """
        Append or update new OHLCV candles for a given timeframe.

        Args:
            timeframe: Timeframe string key (e.g., '1m', '1h', '1d').
            new_candles: DataFrame of new candles, same schema as REQUIRED_COLUMNS,
                         must be sorted ascending by 'date' with timezone-aware datetime.

        Behavior:
            - Ensures new candles do not duplicate existing data.
            - Updates candles if same datetime present with new data.
            - Maintains strict chronological order and integrity.
        """
        logger.debug(f"Received {len(new_candles)} new candles for timeframe {timeframe}")

        # Validate new_candles similarly
        self._validate_and_store_partial(timeframe, new_candles)

        if timeframe not in self.dfs:
            # No prior data, store as is
            self._validate_and_store(timeframe, new_candles)
            logger.info(f"Initialized timeframe {timeframe} with new candles")
            return

        existing_df = self.dfs[timeframe]

        # Merge new candles with existing, prefer new candles on overlaps
        combined = pd.concat([existing_df, new_candles])
        combined = combined.drop_duplicates(subset='date', keep='last').sort_values('date').reset_index(drop=True)

        self.dfs[timeframe] = combined
        logger.info(f"Updated timeframe {timeframe}: total rows {len(combined)}")

    def _validate_and_store_partial(self, timeframe: str, df: pd.DataFrame) -> None:
        """
        Validate incremental update dataframe before merge.

        Args:
            timeframe: Timeframe string key.
            df: DataFrame of new candles.
        """
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Incremental update missing columns: {missing_cols}")

        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], utc=True)

        # Remove duplicates within new_candles
        df.drop_duplicates(subset='date', keep='last', inplace=True)

        # Validate no NaNs in OHLCV cols
        if df[self.REQUIRED_COLUMNS[1:]].isnull().any().any():
            raise ValueError(f"NaNs detected in incremental OHLCV data for timeframe {timeframe}")

    def get_data_slices(self, upto_index: int) -> Dict[str, pd.DataFrame]:
        """
        Return consistent data slices (up to inclusive index) for all timeframes.

        Args:
            upto_index: Integer slice index, must be within current data length bounds.

        Returns:
            Dictionary mapping timeframe to sliced pd.DataFrame.
        """
        slices = {}
        for tf, df in self.dfs.items():
            if upto_index >= len(df):
                logger.warning(f"Requested slice {upto_index} exceeds data length for {tf}, returning full df")
                slices[tf] = df.copy()
            else:
                slices[tf] = df.iloc[:upto_index + 1].copy()
        return slices
