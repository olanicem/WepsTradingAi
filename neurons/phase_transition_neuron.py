#!/usr/bin/env python3
# ==========================================================
# 🔄 WEPS PhaseTransitionNeuron — Final Production Version (Spiral-Aware)
# ✅ Determines Phase Exhaustion & Early New Phase Start
# ✅ Uses Elliott Wave, Candle Patterns & Fibonacci Ratios
# ✅ Outputs Phase Shift Confidence & Timing Factor
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import numpy as np
import logging

logger = logging.getLogger("WEPS.Neurons.PhaseTransitionNeuron")

class PhaseTransitionNeuron:
    """
    WEPS PhaseTransitionNeuron
    - Evaluates if current phase (impulse/corrective) is exhausted.
    - Determines readiness for phase transition using wave depth and pattern signals.
    - Outputs phase shift confidence and timing factor x.
    """
    def __init__(self, elliott_output: dict, candle_output: dict, fibonacci_output: dict, phase: str = "neutral"):
        self.elliott_output = elliott_output
        self.candle_output = candle_output
        self.fibonacci_output = fibonacci_output
        self.phase = phase
        logger.info("PhaseTransitionNeuron initialized with phase=%s", self.phase)

    def compute(self) -> dict:
        # Depth calculation
        elapsed = self.elliott_output.get("elapsed_candles", 0)
        expected = self.elliott_output.get("expected_duration", 50)  # fallback to 50 bars
        x = elapsed / expected if expected > 0 else 0.0

        # Fibonacci overshoot check
        fib_retrace = self.fibonacci_output.get("retracement_ratio", 0.0)
        fib_confidence = 1.0 if 0.38 <= fib_retrace <= 0.618 else 0.0

        # Candle pattern exhaustion
        candle_exhaust = int(self.candle_output.get("exhaustion_detected", False))

        # Combine into phase shift confidence
        phase_shift_conf = np.clip((x + fib_confidence + candle_exhaust) / 3, 0, 1)
        phase_adjusted_conf = round(self._adjust_score_by_phase(phase_shift_conf), 4)

        result = {
            "timing_factor_x": round(x, 4),
            "phase_shift_confidence": phase_adjusted_conf
        }
        logger.info("PhaseTransitionNeuron completed: %s", result)
        return result

    def _adjust_score_by_phase(self, score: float) -> float:
        phase_weights = {
            "rebirth": 1.2,
            "growth": 1.0,
            "decay": 1.2,   # more sensitive to phase shift during decay
            "neutral": 1.0
        }
        adjusted = score * phase_weights.get(self.phase, 1.0)
        logger.debug("Phase-aware phase shift confidence adjusted: base=%.4f, phase=%s, adjusted=%.4f",
                     score, self.phase, adjusted)
        return adjusted
