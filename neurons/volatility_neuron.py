#!/usr/bin/env python3
# ==========================================================
# 🌪️ WEPS VolatilityNeuron — Final Production Version (Phase-Aware)
# ✅ Quantifies Historical & Real-Time Volatility with ATR & Return Variance
# ✅ Identifies Volatility Regime Shifts to Adjust Risk & Reflex Behavior
# ✅ Outputs Phase-Aware Volatility Confidence
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("WEPS.Neurons.VolatilityNeuron")

class VolatilityNeuron:
    """
    WEPS VolatilityNeuron
    - Measures price fluctuation intensity.
    - Identifies volatility regime shifts.
    - Outputs phase-aware volatility confidence.
    """
    def __init__(self, df: pd.DataFrame, phase: str = "neutral"):
        self.df = df
        self.phase = phase
        logger.info("VolatilityNeuron initialized with phase=%s", self.phase)

    def compute(self, df: pd.DataFrame = None) -> dict:
        df = df or self.df
        if df.empty or not all(c in df.columns for c in ["high", "low", "close"]):
            raise ValueError("VolatilityNeuron requires dataframe with 'high', 'low', and 'close' columns.")

        if len(df) < 100:
            raise ValueError("VolatilityNeuron requires at least 100 data points.")

        # Compute ATR
        high, low, close = df['high'], df['low'], df['close']
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]

        # Compute standard deviation of returns
        returns = np.log(close).diff().dropna()
        hist_vol = returns[-50:].std() if len(returns) >= 50 else returns.std()

        # Normalize: relative to last 200 bars
        long_hist_vol = returns[-200:].std() if len(returns) >= 200 else returns.std()
        norm_vol = hist_vol / (long_hist_vol + 1e-9)
        volatility_score = np.clip(norm_vol, 0, 2) / 2  # normalized to [0, 1]

        phase_adjusted = round(np.clip(self._adjust_score_by_phase(volatility_score), 0, 1), 4)

        result = {
            "atr": round(atr, 6),
            "historical_volatility": round(hist_vol, 6),
            "volatility_score_norm": phase_adjusted
        }
        logger.info("VolatilityNeuron completed: %s", result)
        return result

    def _adjust_score_by_phase(self, score: float) -> float:
        phase_weights = {
            "rebirth": 0.8,
            "growth": 1.0,
            "decay": 1.2,
            "neutral": 1.0
        }
        adjusted = score * phase_weights.get(self.phase, 1.0)
        logger.debug("Phase-aware volatility confidence adjusted: base=%.4f, phase=%s, adjusted=%.4f",
                     score, self.phase, adjusted)
        return adjusted
