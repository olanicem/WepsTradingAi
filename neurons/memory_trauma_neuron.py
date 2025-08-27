#!/usr/bin/env python3
# ==========================================================
# 🧠 WEPS MemoryTraumaNeuron — Final Production Version (Phase-Aware)
# ✅ Detects Past Extreme Events & Trauma Memory Factor
# ✅ Measures Recovery Time from Historical Spikes
# ✅ Outputs Phase-Aware Trauma Influence for Reflex Cortex
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("WEPS.Neurons.MemoryTraumaNeuron")

class MemoryTraumaNeuron:
    """
    WEPS MemoryTraumaNeuron
    - Tracks historical extreme price events.
    - Quantifies trauma memory factor & recovery time.
    - Adjusts outputs phase-aware for reflex decision-making.
    """
    def __init__(self, df: pd.DataFrame, phase: str = "neutral"):
        self.df = df
        self.phase = phase
        logger.info("MemoryTraumaNeuron initialized with phase=%s", self.phase)

    def compute(self) -> dict:
        if self.df.empty or "close" not in self.df.columns:
            raise ValueError("MemoryTraumaNeuron requires dataframe with 'close' column.")

        closes = self.df['close'].values
        if len(closes) < 500:
            raise ValueError("MemoryTraumaNeuron requires at least 500 data points for trauma analysis.")

        returns = np.diff(np.log(closes))
        historical_vol = np.std(returns[:-100])
        recent_vol = np.std(returns[-100:])

        # Find largest historical volatility spike
        rolling_vol = pd.Series(returns).rolling(50).std().dropna()
        trauma_spike = rolling_vol.max()
        trauma_factor = np.clip((trauma_spike / (historical_vol + 1e-9)), 0, 5)

        # Estimate recovery time: number of bars needed after trauma spike until volatility normalized
        trauma_idx = rolling_vol.idxmax() if not rolling_vol.empty else 0
        recovery_time = len(returns) - trauma_idx if trauma_idx > 0 else 0

        # Combine into trauma influence score
        trauma_influence = np.tanh((trauma_factor + recovery_time / 1000))
        phase_adjusted = round(np.clip(self._adjust_score_by_phase(trauma_influence), 0, 1), 4)

        result = {
            "trauma_factor": round(trauma_factor, 4),
            "trauma_recovery_time": recovery_time,
            "memory_trauma_score_norm": phase_adjusted
        }
        logger.info("MemoryTraumaNeuron completed: %s", result)
        return result

    def _adjust_score_by_phase(self, score: float) -> float:
        phase_weights = {"rebirth": 0.8, "growth": 0.9, "decay": 1.2, "neutral": 1.0}
        adjusted = score * phase_weights.get(self.phase, 1.0)
        logger.debug("Phase-aware trauma influence adjusted: base=%.4f, phase=%s, adjusted=%.4f",
                     score, self.phase, adjusted)
        return adjusted
