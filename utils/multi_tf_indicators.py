#!/usr/bin/env python3
# ================================================================
# 🧠 WEPS MultiTFIndicators — Institutional-Grade Spiral Intelligence
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Extracts advanced multi-timeframe technical indicators: RSI, MACD hist, ATR, ADX,
#     Bollinger Bands width, Williams Alligator, Ichimoku Cloud, MA200/50, Volume
#   - Computes normalized indicator values plus temporal deltas for trend detection
#   - Implements robust error handling, data validation, and datetime index correction
#   - Designed for seamless integration into WEPS State Vector and RL pipelines
#   - Engineered to surpass highest industry standards in quantitative finance
# ================================================================

import numpy as np
import pandas as pd
import talib
import logging

logger = logging.getLogger("WEPS.MultiTFIndicators")


class MultiTFIndicators:
    def __init__(self, dfs: dict[str, pd.DataFrame], delta_lookback: int = 5):
        """
        :param dfs: Dict[str, pd.DataFrame]
          Keys are timeframes ('1h', '4h', '1d' etc)
          Values are OHLCV dataframes with columns ['open','high','low','close','volume']
        :param delta_lookback: Number of periods to compute temporal delta features for trend capturing.
        """
        self.dfs = dfs or {}
        self.delta_lookback = delta_lookback

    def extract_all(self) -> dict:
        """
        Extracts and aggregates normalized indicator features and deltas for all timeframes.
        Returns a flat dict keyed by '{timeframe}_{indicator}'.
        """
        features = {}
        for tf, df in self.dfs.items():
            try:
                if not self._validate_and_correct_df(tf, df):
                    logger.error(f"[{tf}] Validation failed. Skipping extraction.")
                    continue
                # Assign corrected df back to self.dfs for downstream use
                self.dfs[tf] = df
                tf_features = self.extract_for_tf(tf, df)
                features.update(tf_features)
            except Exception as e:
                logger.error(f"MultiTFIndicators extraction failure for timeframe {tf}: {e}", exc_info=True)
        return features

    def _validate_and_correct_df(self, timeframe: str, df: pd.DataFrame) -> bool:
        """
        Validates presence of required columns, index type, and fixes index if necessary.
        Logs warnings/errors and attempts inplace corrections.
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"[{timeframe}] Missing required columns: {missing_cols}")
            return False

        if df.empty:
            logger.error(f"[{timeframe}] DataFrame is empty.")
            return False

        # Ensure datetime index - attempt conversion from 'date' column if needed
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            logger.warning(f"[{timeframe}] DataFrame index is not datetime. Attempting conversion from 'date' column...")
            if 'date' in df.columns:
                try:
                    df.set_index(pd.to_datetime(df['date']), inplace=True)
                    df.drop(columns=['date'], inplace=True)
                    logger.info(f"[{timeframe}] Converted 'date' column to datetime index.")
                except Exception as e:
                    logger.error(f"[{timeframe}] Failed to convert 'date' column to datetime index: {e}")
                    return False
            else:
                logger.error(f"[{timeframe}] No 'date' column to convert index to datetime.")
                return False

        # Ensure index is monotonic increasing (required for time series methods)
        if not df.index.is_monotonic_increasing:
            logger.warning(f"[{timeframe}] DataFrame index not monotonic increasing. Sorting index.")
            df.sort_index(inplace=True)

        # Check for NaN or infinite values in required columns, warn but allow processing
        for col in required_cols:
            if df[col].isnull().any():
                logger.warning(f"[{timeframe}] Column '{col}' contains NaN values.")
            if not np.isfinite(df[col]).all():
                logger.warning(f"[{timeframe}] Column '{col}' contains infinite or non-finite values.")

        return True

    def extract_for_tf(self, timeframe: str, df: pd.DataFrame) -> dict:
        """
        Extracts normalized technical indicator features and their deltas for the given timeframe.
        """
        features = {}
        if df is None or df.empty or len(df) < self.delta_lookback + 1:
            logger.warning(f"[{timeframe}] Insufficient data to extract features.")
            return features

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        # --- RSI (14) ---
        rsi_series = talib.RSI(close, timeperiod=14)
        rsi = self._safe_last(rsi_series)
        rsi_delta = self._delta(rsi_series, self.delta_lookback)
        features[f"{timeframe}_rsi"] = self._normalize(rsi, 0, 100)
        features[f"{timeframe}_rsi_delta"] = self._scale_delta(rsi_delta, -50, 50)

        # --- MACD Histogram ---
        _, _, macd_hist_series = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        macd_hist = self._safe_last(macd_hist_series)
        macd_hist_delta = self._delta(macd_hist_series, self.delta_lookback)
        features[f"{timeframe}_macd_hist"] = self._normalize(macd_hist, -0.05, 0.05)
        features[f"{timeframe}_macd_hist_delta"] = self._scale_delta(macd_hist_delta, -0.05, 0.05)

        # --- ATR (14) ---
        atr_series = talib.ATR(high, low, close, timeperiod=14)
        atr = self._safe_last(atr_series)
        max_atr = np.nanmax(atr_series) if np.nanmax(atr_series) > 0 else 1
        atr_delta = self._delta(atr_series, self.delta_lookback)
        features[f"{timeframe}_atr"] = self._normalize(atr, 0, max_atr)
        features[f"{timeframe}_atr_delta"] = self._scale_delta(atr_delta, -max_atr, max_atr)

        # --- ADX (14) ---
        adx_series = talib.ADX(high, low, close, timeperiod=14)
        adx = self._safe_last(adx_series)
        adx_delta = self._delta(adx_series, self.delta_lookback)
        features[f"{timeframe}_adx"] = self._normalize(adx, 0, 100)
        features[f"{timeframe}_adx_delta"] = self._scale_delta(adx_delta, -50, 50)

        # --- Bollinger Bands Width ---
        sma = pd.Series(close).rolling(window=20, min_periods=20).mean()
        std = pd.Series(close).rolling(window=20, min_periods=20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        bb_width = ((upper - lower) / sma).iloc[-1] if not pd.isna(sma.iloc[-1]) and sma.iloc[-1] != 0 else 0.0
        bb_width = np.clip(bb_width, 0, 5)
        bb_width_delta = self._delta(pd.Series((upper - lower) / sma), self.delta_lookback)
        features[f"{timeframe}_bb_width"] = bb_width
        features[f"{timeframe}_bb_width_delta"] = self._scale_delta(bb_width_delta, -2.5, 2.5)

        # --- Williams Alligator State ---
        jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().iloc[-1]
        teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().iloc[-1]
        lips = pd.Series(close).rolling(window=5, min_periods=5).mean().iloc[-1]
        alligator_state = self._alligator_state(jaw, teeth, lips)
        features[f"{timeframe}_alligator_uptrend"] = 1.0 if alligator_state == "uptrend" else 0.0
        features[f"{timeframe}_alligator_downtrend"] = 1.0 if alligator_state == "downtrend" else 0.0
        features[f"{timeframe}_alligator_sideways"] = 1.0 if alligator_state == "sideways" else 0.0

        # --- Ichimoku Cloud ---
        ichimoku = self._compute_ichimoku(df)
        for key, val in ichimoku.items():
            features[f"{timeframe}_ichimoku_{key}"] = val

        # --- Moving Averages MA50 and MA200 relative to close ---
        ma50 = pd.Series(close).rolling(window=50, min_periods=50).mean().iloc[-1]
        ma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().iloc[-1]
        features[f"{timeframe}_ma50_vs_close"] = self._normalize(ma50 / close[-1], 0.5, 1.5)
        features[f"{timeframe}_ma200_vs_close"] = self._normalize(ma200 / close[-1], 0.5, 1.5)

        # --- Volume normalized and delta ---
        vol_max = np.max(volume[-20:]) if len(volume) >= 20 else np.max(volume)
        vol_norm = self._normalize(volume[-1], 0, vol_max)
        vol_delta = self._delta(pd.Series(volume), self.delta_lookback)
        features[f"{timeframe}_volume"] = vol_norm
        features[f"{timeframe}_volume_delta"] = self._scale_delta(vol_delta, -vol_max/2, vol_max/2)

        return features

    def _alligator_state(self, jaw: float, teeth: float, lips: float) -> str:
        if lips > teeth > jaw:
            return "uptrend"
        elif lips < teeth < jaw:
            return "downtrend"
        else:
            return "sideways"

    def _compute_ichimoku(self, df: pd.DataFrame) -> dict:
        high = df['high']
        low = df['low']
        close = df['close']

        period9_high = high.rolling(window=9, min_periods=9).max()
        period9_low = low.rolling(window=9, min_periods=9).min()
        tenkan_sen = (period9_high + period9_low) / 2

        period26_high = high.rolling(window=26, min_periods=26).max()
        period26_low = low.rolling(window=26, min_periods=26).min()
        kijun_sen = (period26_high + period26_low) / 2

        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_span_b = ((high.rolling(window=52, min_periods=52).max() + low.rolling(window=52, min_periods=52).min()) / 2).shift(26)

        chikou_span = close.shift(-26)

        last_close = close.iloc[-1]

        components = {
            'tenkan_sen': self._normalize(tenkan_sen.iloc[-1], 0, 2 * last_close),
            'kijun_sen': self._normalize(kijun_sen.iloc[-1], 0, 2 * last_close),
            'senkou_span_a': self._normalize(senkou_span_a.iloc[-1], 0, 2 * last_close),
            'senkou_span_b': self._normalize(senkou_span_b.iloc[-1], 0, 2 * last_close),
            'chikou_span': self._normalize(chikou_span.iloc[-1], 0, 2 * last_close),
            'cloud_bias': 1.0 if last_close > max(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1]) else 0.0
        }
        return components

    def _delta(self, series: pd.Series, lookback: int) -> float:
        if series is None or len(series) <= lookback:
            return 0.0
        try:
            return float(series.iloc[-1] - series.iloc[-1 - lookback])
        except Exception as e:
            logger.debug(f"Delta calc failed: {e}")
            return 0.0

    def _safe_last(self, series) -> float:
        if series is None or len(series) == 0:
            return 0.0
        for val in reversed(series):
            if not np.isnan(val):
                return float(val)
        return 0.0

    def _normalize(self, val: float, min_val: float, max_val: float) -> float:
        if val is None or np.isnan(val):
            return 0.0
        norm = (val - min_val) / (max_val - min_val + 1e-8)
        return min(max(norm, 0.0), 1.0)

    def _scale_delta(self, delta_val: float, min_val: float, max_val: float) -> float:
        if delta_val is None or np.isnan(delta_val):
            return 0.5
        scaled = (delta_val - min_val) / (max_val - min_val + 1e-8)
        return min(max(scaled, 0.0), 1.0)
