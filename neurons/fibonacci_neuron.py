#!/usr/bin/env python3
# ==========================================================
# 🌀 WEPS FibonacciNeuron v4.0 — Supreme Spiral Intelligence Edition
# ✅ Advanced wave order validation
# ✅ Confluence with momentum & cycle signals
# ✅ Spiral phase-aware confidence with safety checks
# ✅ Institutional-grade precision & logging
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import numpy as np
import logging

logger = logging.getLogger("WEPS.Neurons.FibonacciNeuron")

class FibonacciNeuron:
    """
    WEPS FibonacciNeuron v4.0
    - Validates Elliott wave displacement accuracy.
    - Computes expected vs observed retracement & timing.
    - Uses spiral intelligence to refine confidence.
    """

    def __init__(self, df, phase: str = "neutral"):
        self.df = df
        self.phase = phase
        logger.info("FibonacciNeuron v4.0 initialized with phase=%s", self.phase)

    def compute(self, elliott_output: dict, cycle_data: dict = None, momentum_data: dict = None) -> dict:
        closes = self.df['close'].values

        impulse_waves = elliott_output.get('impulse_waves', [])
        corrective_waves = elliott_output.get('corrective_waves', [])
        logger.debug("Elliott impulse_waves: %s", impulse_waves)
        logger.debug("Elliott corrective_waves: %s", corrective_waves)

        # Validate waves
        if len(impulse_waves) < 5 or len(corrective_waves) < 3:
            raise ValueError("Insufficient Elliott wave structure: expected 5 impulse and 3 corrective waves.")

        impulse_start_idx = impulse_waves[0]['index']
        impulse_end_idx = impulse_waves[-1]['index']
        corrective_extreme_idx = corrective_waves[-1]['index']

        # Validate index order
        if not (0 <= impulse_start_idx < impulse_end_idx < corrective_extreme_idx < len(closes)):
            logger.error("Wave index inconsistency: start=%d, end=%d, corrective=%d, closes_length=%d",
                         impulse_start_idx, impulse_end_idx, corrective_extreme_idx, len(closes))
            raise ValueError("Invalid wave order or indices exceed data length.")

        impulse_range = abs(closes[impulse_end_idx] - closes[impulse_start_idx])
        if impulse_range < 1e-6:
            logger.warning("Impulse range too small: %.8f", impulse_range)
            raise ValueError("Impulse range too small for meaningful displacement calculation.")

        # Compute observed displacement ratio
        observed_displacement = abs(closes[corrective_extreme_idx] - closes[impulse_end_idx]) / impulse_range
        observed_displacement = np.clip(observed_displacement, 0, 2)

        expected_displacement = 0.382 if self.phase in ["rebirth", "growth"] else 0.618

        displacement_score = 1 - abs((observed_displacement - expected_displacement) / (expected_displacement + 1e-9))
        displacement_score = np.clip(displacement_score, 0.0, 1.0)

        impulse_duration = impulse_end_idx - impulse_start_idx
        expected_corrective_duration = impulse_duration * 0.63
        actual_corrective_duration = corrective_extreme_idx - impulse_end_idx
        timing_factor = max(0.0, actual_corrective_duration / (expected_corrective_duration + 1e-9))

        # Integrate spiral intelligence
        spiral_boost = 1.0
        if cycle_data and cycle_data.get("cycle_confidence", 0.0) > 0.8:
            spiral_boost *= 1.05
        if momentum_data and momentum_data.get("momentum_score_norm", 0.0) > 0.5:
            spiral_boost *= 1.05

        phase_confidence = np.clip(self._adjust_confidence_by_phase(displacement_score) * spiral_boost, 0, 1)

        result = {
            'expected_displacement': round(expected_displacement, 4),
            'observed_displacement': round(observed_displacement, 4),
            'displacement_score': round(displacement_score, 4),
            'timing_factor': round(timing_factor, 4),
            'phase_aware_confidence': round(phase_confidence, 4)
        }

        logger.info("FibonacciNeuron v4.0 completed: %s", result)
        return result

    def _adjust_confidence_by_phase(self, score: float) -> float:
        """Adjust displacement confidence by spiral phase context."""
        phase_weights = {
            "rebirth": 1.2,
            "growth": 1.0,
            "decay": 0.8,
            "neutral": 1.0
        }
        adjusted = score * phase_weights.get(self.phase, 1.0)
        logger.debug("Phase-aware confidence adjusted: base=%.4f, phase=%s, adjusted=%.4f",
                     score, self.phase, adjusted)
        return adjusted
