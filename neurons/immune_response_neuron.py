#!/usr/bin/env python3
# ==========================================================
# 🛡️ WEPS ImmuneResponseNeuron — Hardened Spiral Intelligence Version
# ✅ Monitors internal resilience & integrates SentimentNeuron safely
# ✅ Detects abnormal resilience shifts or phase-violating behavior
# ✅ Advises defensive posture proactively, even on malformed sentiment input
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import numpy as np
import logging

logger = logging.getLogger("WEPS.Neurons.ImmuneResponseNeuron")

class ImmuneResponseNeuron:
    """
    WEPS ImmuneResponseNeuron
    - Aggregates internal and external signals.
    - Detects phase-violating behavior triggered by risk-on/off sentiment.
    - Now robustly validates sentiment_output on init to prevent pipeline crashes.
    """

    def __init__(self, state_outputs: dict, sentiment_output, phase: str = "neutral"):
        self.state_outputs = state_outputs
        # ✅ Hardened sentiment_output validation
        if isinstance(sentiment_output, dict):
            self.sentiment_output = sentiment_output
        else:
            logger.warning("[ImmuneResponseNeuron] Invalid sentiment_output type: %s; using fallback dict.", type(sentiment_output))
            self.sentiment_output = {"sentiment_adjustment": 0.0, "sentiment_signal": "neutral"}
        self.phase = phase
        logger.info("ImmuneResponseNeuron initialized with phase=%s", self.phase)

    def compute(self) -> dict:
        try:
            # 🩸 Aggregate internal resilience signals
            indicator_anomaly = abs(self._safe_get("indicator", "long_ema_slope_norm"))
            fractal_collapse = int(self._safe_get("fractal", "fractal_collapse"))
            half_life_short = max(0.0, 1 - self._safe_get("half_life", "half_life_score_norm"))
            volatility_surge = abs(self._safe_get("volatility", "volatility_score_norm") - 0.5) * 2
            entropy_spike = abs(self._safe_get("entropy", "entropy_score", fallback=0.0) - 0.5) * 2

            destruction_index = np.clip(
                np.mean([indicator_anomaly, fractal_collapse, half_life_short, volatility_surge, entropy_spike]),
                0, 1
            )

            # 🎯 Integrate sentiment risk-on/off adjustment
            sentiment_bias = self.sentiment_output.get("sentiment_adjustment", 0.0)
            sentiment_signal = self.sentiment_output.get("sentiment_signal", "neutral")

            adjusted_destruction = destruction_index - (
                sentiment_bias * 0.5 if sentiment_signal == "risk_on" else -sentiment_bias * 0.5
            )
            immunity_confidence = round(np.clip(self._adjust_score_by_phase(1 - adjusted_destruction), 0, 1), 4)

            caution_note = self._generate_defensive_note(
                destruction_index, sentiment_signal, self.phase, immunity_confidence
            )

            result = {
                "destruction_index": round(destruction_index, 4),
                "immunity_confidence": immunity_confidence,
                "early_exit_signal": destruction_index > 0.75,
                "sentiment_signal": sentiment_signal,
                "caution_note": caution_note
            }
            logger.info("ImmuneResponseNeuron completed: %s", result)
            return result

        except Exception as e:
            logger.error("ImmuneResponseNeuron compute() failed: %s", e, exc_info=True)
            return {
                "destruction_index": 1.0,
                "immunity_confidence": 0.0,
                "early_exit_signal": True,
                "sentiment_signal": "error",
                "caution_note": "🚨 Critical error in risk defense; immediate caution advised."
            }

    def _safe_get(self, neuron: str, key: str, fallback=0.0):
        try:
            neuron_data = self.state_outputs.get(neuron, {})
            if isinstance(neuron_data, dict):
                return neuron_data.get(key, fallback)
            else:
                logger.warning("[ImmuneResponseNeuron] neuron=%s returned invalid type: %s; using fallback=%s",
                               neuron, type(neuron_data), fallback)
                return fallback
        except Exception as e:
            logger.error("[ImmuneResponseNeuron] _safe_get() failed for neuron=%s key=%s: %s",
                         neuron, key, e, exc_info=True)
            return fallback

    def _adjust_score_by_phase(self, score: float) -> float:
        phase_weights = {
            "rebirth": 1.2,
            "growth": 1.0,
            "decay": 0.8,
            "neutral": 1.0
        }
        adjusted = score * phase_weights.get(self.phase, 1.0)
        logger.debug("Phase-aware immunity confidence adjusted: base=%.4f, phase=%s, adjusted=%.4f",
                     score, self.phase, adjusted)
        return adjusted

    def _generate_defensive_note(self, destruction_index, sentiment_signal, phase, immunity_conf):
        if phase in ["rebirth", "growth"] and destruction_index > 0.7 and sentiment_signal == "risk_off":
            return "🚨 Strong risk-off detected during impulse; consider defensive short or early exit."
        elif phase == "decay" and destruction_index < 0.3 and sentiment_signal == "risk_on":
            return "🚨 Unusual strength surge detected in decay; possible corrective reversal favoring long."
        elif immunity_conf < 0.5:
            return "⚠️ Low immunity confidence; caution advised before confirming trade."
        else:
            return "✅ Immunity stable; trade confidence aligned with market resilience."
