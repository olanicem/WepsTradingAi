#!/usr/bin/env python3
# ==========================================================
# 🔄 WEPS CycleNeuron v2.0 — Institutional Spiral Intelligence
# ✅ Measures Real-Time Wave Cycle Progression
# ✅ Detects Main/Nested Cycles with Amplitude, Duration, Momentum
# ✅ Outputs Phase-Aware Cycle Completion Confidence
# ✅ Defensive DataFrame Validation with Citadel-Grade Engineering
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("WEPS.Neurons.CycleNeuronV2")

class CycleNeuron:
    """
    WEPS Spiral CycleNeuron v2.0
    - Tracks Elliott-like wave cycles: impulse vs corrective.
    - Measures amplitude, duration, momentum.
    - Outputs phase-aware completion probability.
    """

    def __init__(self, df: pd.DataFrame, phase: str = "neutral"):
        self.df = df
        self.phase = phase
        logger.info("CycleNeuron initialized with phase=%s", self.phase)

    def compute(self, df: pd.DataFrame = None) -> dict:
        if df is None:
            df = self.df
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"CycleNeuron: Provided df is not a DataFrame (got {type(df)}).")
        if df.empty or "close" not in df.columns:
            raise ValueError("CycleNeuron requires dataframe with 'close' column and non-empty data.")

        closes = df['close'].values
        if len(closes) < 50:
            raise ValueError("CycleNeuron requires at least 50 data points.")

        # Calculate amplitude & duration since local min/max (as wave start proxy)
        recent_min_idx = np.argmin(closes[-50:])
        recent_max_idx = np.argmax(closes[-50:])
        wave_start_idx = min(recent_min_idx, recent_max_idx)
        amplitude = abs(closes[-1] - closes[-50 + wave_start_idx])
        duration = len(closes) - (len(closes) - 50 + wave_start_idx)
        momentum = np.mean(np.diff(closes[-duration:])) if duration > 0 else 0.0

        # Expected duration from prior impulse (approx 63% ratio)
        expected_duration = max(1, int(duration / 0.63))
        completion_factor = round(min(duration / expected_duration, 1.5), 4)

        # Determine if current cycle looks impulsive or corrective
        cycle_type = "impulsive" if momentum * amplitude > 0 else "corrective"

        # Adjust cycle confidence by phase context
        base_confidence = 1 - abs(1 - completion_factor)
        cycle_confidence = round(self._adjust_confidence_by_phase(base_confidence), 4)

        result = {
            "current_cycle_type": cycle_type,
            "cycle_amplitude": round(amplitude, 4),
            "cycle_duration": duration,
            "cycle_completion_factor": completion_factor,
            "cycle_confidence": cycle_confidence
        }
        logger.info("CycleNeuron completed: %s", result)
        return result

    def _adjust_confidence_by_phase(self, score: float) -> float:
        phase_weights = {"rebirth": 1.2, "growth": 1.0, "decay": 0.8, "neutral": 1.0}
        adjusted = np.clip(score * phase_weights.get(self.phase, 1.0), 0, 1)
        logger.debug("Phase-adjusted cycle confidence: base=%.4f, phase=%s, adjusted=%.4f",
                     score, self.phase, adjusted)
        return adjusted
