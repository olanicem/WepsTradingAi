#!/usr/bin/env python3
# ================================================================
# 🔬 WEPS TrendNeuron v5.1 — Spiral-Compliant Institutional Model
# ✅ Full Trend Verification, Exhaustion, and Wave-Phase Alignment
# ✅ Uses EMA Slope Curvature, ADX Strength, Fractal-Wave Harmony
# ✅ Rejects False Trends, Detects Emerging or Dying Trends
# ✅ Phase-Aware Confidence, Biometric Grade Logging, Reflex Sync
# Author: Ola Bode (WEPS Creator)
# ================================================================
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("WEPS.Neurons.TrendNeuron")

class TrendNeuron:
    def __init__(self, df: pd.DataFrame, phase: str = "neutral"):
        self.df = df.copy()
        self.phase = phase
        logger.info("[TrendNeuron] Initialized | Phase=%s | Length=%d", self.phase, len(df))

    def compute(self, df: pd.DataFrame = None) -> dict:
        df = df or self.df
        if df.empty or 'close' not in df.columns:
            raise ValueError("TrendNeuron requires dataframe with 'close' column.")

        df["ema21"] = df["close"].ewm(span=21).mean()
        df["ema50"] = df["close"].ewm(span=50).mean()
        df["ema200"] = df["close"].ewm(span=200).mean()

        # --- 1. EMA Slope Analysis (Trend Vector)
        slope_21 = self._slope(df["ema21"].iloc[-21:])
        slope_50 = self._slope(df["ema50"].iloc[-50:])
        slope_200 = self._slope(df["ema200"].iloc[-50:])

        direction = self._trend_direction(slope_21, slope_50, slope_200)
        alignment_score = self._alignment_score(slope_21, slope_50, slope_200)
        trend_strength_raw = np.tanh(abs(slope_21) + abs(slope_50) + abs(slope_200))

        # --- 2. Exhaustion Detection via Momentum Decay
        ema_diff = df["ema21"] - df["ema50"]
        decay = np.std(ema_diff.iloc[-20:]) / (np.mean(np.abs(ema_diff.iloc[-20:])) + 1e-6)
        exhaustion_score = 1.0 - np.clip(decay, 0, 1)

        # --- 3. ADX-based Trend Filter
        adx = self._compute_adx(df)
        latest_adx = adx.iloc[-1]
        trend_valid = int(latest_adx >= 20 and alignment_score >= 0.7)

        # --- 4. Final Normalized Score (FESI Compliant)
        raw_score = trend_strength_raw * exhaustion_score * trend_valid
        phase_multiplier = {
            "rebirth": 1.2,
            "growth": 1.0,
            "decay": 0.5,
            "death": 0.3,
            "neutral": 0.8
        }
        final_score = round(np.clip(raw_score * phase_multiplier.get(self.phase, 1.0), 0, 1), 4)

        result = {
            "trend_direction": direction,
            "trend_strength": round(trend_strength_raw, 4),
            "exhaustion_score": round(exhaustion_score, 4),
            "alignment_score": round(alignment_score, 4),
            "adx": round(latest_adx, 2),
            "valid_trend": bool(trend_valid),
            "trend_momentum_norm": final_score
        }

        logger.info("[TrendNeuron] Completed: %s", result)
        return result

    def _slope(self, series: pd.Series) -> float:
        """Returns slope of a time series using linear regression."""
        x = np.arange(len(series))
        y = series.values
        return np.polyfit(x, y, 1)[0]

    def _alignment_score(self, s21, s50, s200) -> float:
        """Measures if all slopes point in same direction."""
        signs = np.sign([s21, s50, s200])
        same_sign = int(len(set(signs)) == 1)
        average_magnitude = np.mean([abs(s21), abs(s50), abs(s200)])
        return same_sign * min(1.0, average_magnitude * 50)

    def _trend_direction(self, s21, s50, s200) -> str:
        if all(s > 0 for s in [s21, s50, s200]):
            return "up"
        elif all(s < 0 for s in [s21, s50, s200]):
            return "down"
        return "neutral"

    def _compute_adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = -minus_dm

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window).mean()
        plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-6)
        return dx.rolling(window).mean().fillna(0)
