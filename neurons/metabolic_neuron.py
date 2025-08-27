#!/usr/bin/env python3
# ==========================================================
# ⚡ WEPS MetabolicNeuron — Final Production Version (Phase-Aware)
# ✅ Measures Asset’s “Metabolism” via Volume & Momentum Acceleration
# ✅ Quantifies Internal Energy Flow & Market Participation
# ✅ Outputs Phase-Aware Metabolic Rate Index (MRI)
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("WEPS.Neurons.MetabolicNeuron")

class MetabolicNeuron:
    """
    WEPS MetabolicNeuron
    - Models trading activity acceleration as financial metabolism.
    - Outputs MRI: internal energy score aligned with spiral phases.
    """
    def __init__(self, df: pd.DataFrame, phase: str = "neutral"):
        self.df = df
        self.phase = phase
        logger.info("MetabolicNeuron initialized with phase=%s", self.phase)

    def compute(self) -> dict:
        if self.df.empty or not all(c in self.df.columns for c in ["close", "volume"]):
            raise ValueError("MetabolicNeuron requires dataframe with 'close' and 'volume' columns.")

        closes, volumes = self.df['close'].values, self.df['volume'].values
        if len(closes) < 100:
            raise ValueError("MetabolicNeuron requires at least 100 data points.")

        returns = np.diff(np.log(closes))
        momentum_acceleration = np.mean(np.abs(np.diff(returns[-50:])))
        volume_acceleration = np.mean(np.abs(np.diff(volumes[-50:]))) / np.mean(volumes[-100:])

        # Normalize both to 0-1 scale:
        momentum_score = np.tanh(momentum_acceleration * 100)
        volume_score = np.tanh(volume_acceleration * 10)

        # Combine into MRI:
        metabolic_rate_index = np.clip(0.6 * momentum_score + 0.4 * volume_score, 0, 1)
        phase_adjusted = round(np.clip(self._adjust_score_by_phase(metabolic_rate_index), 0, 1), 4)

        result = {
            "metabolism_score": phase_adjusted
        }
        logger.info("MetabolicNeuron completed: %s", result)
        return result

    def _adjust_score_by_phase(self, score: float) -> float:
        phase_weights = {
            "rebirth": 1.2,   # rising metabolism supports new trends
            "growth": 1.0,
            "decay": 0.8,     # high metabolism can signal trap in decay
            "neutral": 1.0
        }
        adjusted = score * phase_weights.get(self.phase, 1.0)
        logger.debug("Phase-aware metabolism adjusted: base=%.4f, phase=%s, adjusted=%.4f",
                     score, self.phase, adjusted)
        return adjusted
