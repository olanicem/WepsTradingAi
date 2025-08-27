#!/usr/bin/env python3
# ==========================================================
# 🔮 WEPS FractalNeuron — Final Production Version (Phase-Aware)
# ✅ Detects Nested Fractal Alignment via Volatility & Entropy
# ✅ Computes Fractal Entropy Index (FEI) and Coherence Score
# ✅ Outputs Phase-Aware Fractal Confidence
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("WEPS.Neurons.FractalNeuron")

class FractalNeuron:
    """
    WEPS FractalNeuron
    - Measures nested fractals (main, sub, micro) to detect fractal coherence.
    - Quantifies harmony vs. chaos using Fractal Entropy Index (FEI).
    - Outputs phase-aware fractal confidence for WEPS decisions.
    """
    def __init__(self, df: pd.DataFrame, phase: str = "neutral"):
        self.df = df
        self.phase = phase
        logger.info("FractalNeuron initialized with phase=%s", self.phase)

    def compute(self, df: pd.DataFrame = None) -> dict:
        df = df or self.df
        if df.empty or "close" not in df.columns:
            raise ValueError("FractalNeuron requires dataframe with 'close' column.")

        closes = df['close'].values
        if len(closes) < 300:
            raise ValueError("FractalNeuron requires at least 300 data points.")

        # Main fractal volatility: ~200-bar window
        main_returns = np.diff(np.log(closes[-200:]))
        main_vol = np.std(main_returns) + 1e-9

        # Sub fractal volatility: ~100-bar window
        sub_returns = np.diff(np.log(closes[-100:]))
        sub_vol = np.std(sub_returns) + 1e-9

        # Micro fractal volatility: ~50-bar window
        micro_returns = np.diff(np.log(closes[-50:]))
        micro_vol = np.std(micro_returns) + 1e-9

        # Compute Fractal Entropy Index (FEI) from nested volatilities
        fei = (micro_vol / (sub_vol + 1e-9)) * (sub_vol / (main_vol + 1e-9))
        fei = np.clip(fei, 0, 10)  # Prevent extreme values

        # Compute Fractal Coherence Score: lower FEI → higher coherence
        coherence = 1.0 - np.clip(fei, 0, 1)

        # Adjust coherence by spiral phase
        phase_adjusted_coherence = round(self._adjust_score_by_phase(coherence), 4)

        result = {
            "fei": round(fei, 4),
            "fractal_coherence_score": round(coherence, 4),
            "phase_aware_fractal_confidence": phase_adjusted_coherence
        }
        logger.info("FractalNeuron completed: %s", result)
        return result

    def _adjust_score_by_phase(self, score: float) -> float:
        """Scale coherence confidence by current spiral phase."""
        phase_weights = {
            "rebirth": 1.2,
            "growth": 1.0,
            "decay": 0.8,
            "neutral": 1.0
        }
        adjusted = score * phase_weights.get(self.phase, 1.0)
        logger.debug("Phase-aware fractal confidence adjusted: base=%.4f, phase=%s, adjusted=%.4f",
                     score, self.phase, adjusted)
        return adjusted
