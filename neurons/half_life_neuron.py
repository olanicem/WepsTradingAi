#!/usr/bin/env python3
# ==========================================================
# ⏳ WEPS HalfLifeNeuron — Final Production Version (Phase-Aware)
# ✅ Computes Half-Life of Momentum & Volatility
# ✅ Quantifies Trend & Volatility Decay Timing
# ✅ Outputs Phase-Aware Half-Life Score
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("WEPS.Neurons.HalfLifeNeuron")

class HalfLifeNeuron:
    """
    WEPS HalfLifeNeuron
    - Measures how quickly momentum and volatility decay (half-life).
    - Quantifies trend persistence and time-to-exhaustion.
    - Outputs phase-aware half-life score for spiral trading decisions.
    """
    def __init__(self, df: pd.DataFrame, phase: str = "neutral"):
        self.df = df
        self.phase = phase
        logger.info("HalfLifeNeuron initialized with phase=%s", self.phase)

    def compute(self, df: pd.DataFrame = None) -> dict:
        df = df or self.df
        if df.empty or "close" not in df.columns:
            raise ValueError("HalfLifeNeuron requires dataframe with 'close' column.")

        closes = df['close'].values
        if len(closes) < 200:
            raise ValueError("HalfLifeNeuron requires at least 200 data points.")

        # Compute momentum half-life using returns decay
        returns = np.diff(np.log(closes))
        abs_returns = np.abs(returns[-100:])
        if np.all(abs_returns == 0):
            momentum_half_life = 100.0
        else:
            peak = np.max(abs_returns)
            half_peak = peak / 2.0
            try:
                idx = np.where(abs_returns <= half_peak)[0][0]
                momentum_half_life = idx
            except IndexError:
                momentum_half_life = len(abs_returns)

        # Compute volatility half-life after recent spike
        rolling_vol = pd.Series(abs_returns).rolling(20).std().fillna(0)
        recent_vol = rolling_vol.iloc[-1]
        half_vol = recent_vol / 2.0
        idx_vol_decay = np.where(rolling_vol[::-1] <= half_vol)[0]
        volatility_half_life = idx_vol_decay[0] if len(idx_vol_decay) else len(rolling_vol)

        avg_half_life = np.mean([momentum_half_life, volatility_half_life])
        phase_adjusted = round(self._adjust_score_by_phase(avg_half_life / 100), 4)

        result = {
            "momentum_half_life": int(momentum_half_life),
            "volatility_half_life": int(volatility_half_life),
            "half_life_score_norm": phase_adjusted
        }
        logger.info("HalfLifeNeuron completed: %s", result)
        return result

    def _adjust_score_by_phase(self, score: float) -> float:
        """Scale half-life confidence by current spiral phase."""
        phase_weights = {
            "rebirth": 1.2,
            "growth": 1.0,
            "decay": 0.8,
            "neutral": 1.0
        }
        adjusted = score * phase_weights.get(self.phase, 1.0)
        logger.debug("Phase-aware half-life score adjusted: base=%.4f, phase=%s, adjusted=%.4f",
                     score, self.phase, adjusted)
        return adjusted
