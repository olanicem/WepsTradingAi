#!/usr/bin/env python3
# ==========================================================
# 💧 WEPS LiquidityNeuron — Final Production Version (Phase-Aware)
# ✅ Measures Liquidity Surge or Contraction with Historical Baselines
# ✅ Computes Stability of Liquidity (Volume Consistency)
# ✅ Outputs Phase-Aware Liquidity Confidence
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("WEPS.Neurons.LiquidityNeuron")


class LiquidityNeuron:
    """
    WEPS LiquidityNeuron
    - Quantifies liquidity surge/drought relative to historical averages.
    - Measures stability of liquidity (volume variability).
    - Outputs normalized, phase-aware liquidity confidence.
    """

    def __init__(self, df: pd.DataFrame, phase: str = "neutral"):
        self.df = df
        self.phase = phase
        logger.info("LiquidityNeuron initialized with phase=%s", self.phase)

    def compute(self, df: pd.DataFrame = None) -> dict:
        df = df or self.df
        if df.empty or "volume" not in df.columns:
            raise ValueError("LiquidityNeuron requires dataframe with 'volume' column.")

        volumes = df['volume'].values
        if len(volumes) < 250:
            raise ValueError("LiquidityNeuron requires at least 250 data points.")

        recent_vol = np.mean(volumes[-20:])
        hist_vol = np.mean(volumes[-250:])
        liquidity_score = recent_vol / (hist_vol + 1e-9)

        stability = np.std(volumes[-20:]) / (hist_vol + 1e-9)
        phase_adjusted = round(np.clip(self._adjust_score_by_phase(liquidity_score), 0, 2), 4)

        result = {
            "liquidity_score_norm": phase_adjusted,
            "liquidity_stability": round(stability, 4)
        }
        logger.info("LiquidityNeuron completed: %s", result)
        return result

    def _adjust_score_by_phase(self, score: float) -> float:
        """Scales liquidity confidence based on current spiral phase."""
        phase_weights = {
            "rebirth": 1.2,   # liquidity re-emerging → boost confidence
            "growth": 1.0,    # stable liquidity expected
            "decay": 0.8,     # falling liquidity → caution
            "neutral": 1.0
        }
        adjusted = score * phase_weights.get(self.phase, 1.0)
        logger.debug("Phase-aware liquidity score adjusted: base=%.4f, phase=%s, adjusted=%.4f",
                     score, self.phase, adjusted)
        return adjusted
