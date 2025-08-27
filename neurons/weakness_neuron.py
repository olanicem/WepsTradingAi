#!/usr/bin/env python3
# ==========================================================
# ⚠️ WEPS WeaknessNeuron v2.0 — Spiral-Confirmed
# ✅ Works with confirmed Elliott waves from SpiralWaveEngine.
# ✅ Early exit with clear reason on invalid impulse.
# ✅ Integrates phase-weighted confidence.
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import numpy as np
import logging

logger = logging.getLogger("WEPS.Neurons.WeaknessNeuron")

class WeaknessNeuron:
    """
    WEPS WeaknessNeuron v2.0
    - Confirms corrective wave weakness relative to validated impulse.
    - Leverages phase confidence for more robust signals.
    """
    def __init__(self, elliott_output: dict, phase: str = "neutral"):
        self.elliott_output = elliott_output
        self.phase = phase
        logger.info("WeaknessNeuron initialized with phase=%s", self.phase)

    def compute(self) -> dict:
        if not self.elliott_output.get("valid_impulse", False):
            logger.warning("WeaknessNeuron: invalid impulse detected, skipping weakness analysis.")
            return {"weakness_score": 0.0, "reversal_warning": False, "reason": "invalid_impulse"}

        impulse = self.elliott_output.get("confirmed_impulse", [])
        corrective = self.elliott_output.get("confirmed_corrective", [])

        if len(impulse) < 2 or len(corrective) < 2:
            logger.warning("WeaknessNeuron: insufficient confirmed waves for reliable analysis.")
            return {"weakness_score": 0.0, "reversal_warning": False, "reason": "too_few_waves"}

        impulse_size = abs(impulse[-1]['price'] - impulse[-2]['price'])
        corrective_size = abs(corrective[-1]['price'] - corrective[-2]['price'])

        size_ratio = corrective_size / (impulse_size + 1e-9)  # avoid division by zero
        duration_ratio = (corrective[-1]['index'] - corrective[-2]['index']) / \
                         max(impulse[-1]['index'] - impulse[-2]['index'], 1)

        combined_weakness = np.tanh(size_ratio + duration_ratio)
        phase_confidence = self.elliott_output.get("wave_confidence", 0.0)

        final_weakness = np.clip(combined_weakness * (1.0 + phase_confidence), 0, 1)
        final_weakness = round(self._adjust_score_by_phase(final_weakness), 4)

        reversal_warning = final_weakness > 0.7  # threshold

        result = {
            "weakness_score": final_weakness,
            "reversal_warning": reversal_warning,
            "reason": "ok"
        }
        logger.info("WeaknessNeuron completed: %s", result)
        return result

    def _adjust_score_by_phase(self, score: float) -> float:
        phase_weights = {
            "rebirth": 0.7,
            "growth": 0.5,
            "decay": 1.3,
            "neutral": 1.0
        }
        adjusted = score * phase_weights.get(self.phase, 1.0)
        logger.debug("Phase-adjusted weakness score: base=%.4f, phase=%s, adjusted=%.4f",
                     score, self.phase, adjusted)
        return adjusted
