#!/usr/bin/env python3
# ==========================================================
# 🌀 WEPS SpiralDensityNeuron — Final Production Version
# ✅ Measures Candle Density & Compression/Expansion Dynamics
# ✅ Detects Energy Buildup for Breakouts or Phase Exhaustion
# ✅ Outputs Phase-Aware Spiral Density Score
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("WEPS.Neurons.SpiralDensityNeuron")

class SpiralDensityNeuron:
    """
    WEPS SpiralDensityNeuron
    - Quantifies candle density in recent price action.
    - Detects compression (coiling) or expansion patterns within spiral phases.
    - Outputs normalized, phase-aware spiral density score.
    """
    def __init__(self, df: pd.DataFrame, phase: str = "neutral", window: int = 50):
        self.df = df
        self.phase = phase
        self.window = window
        logger.info("SpiralDensityNeuron initialized with phase=%s", self.phase)

    def compute(self, df: pd.DataFrame = None) -> dict:
        df = df or self.df
        if df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
            raise ValueError("SpiralDensityNeuron requires dataframe with 'high', 'low', and 'close' columns.")

        if len(df) < self.window:
            raise ValueError(f"SpiralDensityNeuron requires at least {self.window} data points.")

        recent = df.tail(self.window)
        true_ranges = recent['high'] - recent['low']
        atr = np.mean(true_ranges)
        close_std_dev = np.std(recent['close'])
        density_ratio = atr / (close_std_dev + 1e-9)

        phase_adjusted = round(np.clip(self._adjust_score_by_phase(density_ratio / 5), 0, 1), 4)

        result = {
            "spiral_density_score_norm": phase_adjusted
        }
        logger.info("SpiralDensityNeuron completed: %s", result)
        return result

    def _adjust_score_by_phase(self, score: float) -> float:
        phase_weights = {"rebirth": 1.2, "growth": 1.0, "decay": 0.8, "neutral": 1.0}
        return score * phase_weights.get(self.phase, 1.0)
