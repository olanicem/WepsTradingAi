#!/usr/bin/env python3
# ==========================================================
# ⚡ WEPS ImpulseNeuron — Full Production Grade FESI-Compliant Spiral Intelligence Core
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Multi-dimensional impulse strength evaluation using fractal wave structure,
#     momentum dynamics, and metabolic energy metaphors
#   - Adaptive phase weighting based on spiral entropy & mutation signals
#   - Volume/liquidity-informed impulse reliability estimation
#   - Sentiment and macro-sync context integration for reflexive market state awareness
#   - Uncertainty quantification via impulse variance and confidence intervals
#   - Defensive checks for input validity with detailed logging
# ==========================================================
import numpy as np
import logging

logger = logging.getLogger("WEPS.Neurons.ImpulseNeuron")

class ImpulseNeuron:
    def __init__(self, elliott_wave_output, volume_series=None,
                 sentiment_signal=0.0, macro_sync_state=0.5,
                 spiral_entropy=0.5, mutation_level=0.0, phase="neutral"):
        """
        Args:
            elliott_wave_output (dict): Output from ElliottWaveNeuron including:
                - 'impulse_waves': list of dicts with 'price', 'time' keys
                - 'wave_probability_score' (float)
                - 'wave_confidence' (float)
            volume_series (np.ndarray or list): Recent volume data for liquidity context
            sentiment_signal (float): Sentiment influence normalized [-1,1]
            macro_sync_state (float): Market regime synchronization [0,1]
            spiral_entropy (float): Spiral entropy metric [0,1]
            mutation_level (float): Market mutation score [0,1]
            phase (str): Spiral phase ("rebirth", "growth", "decay", "neutral", "death")
        """
        if elliott_wave_output is None or (hasattr(elliott_wave_output, 'empty') and elliott_wave_output.empty):
            self.ew_output = {}
        else:
            self.ew_output = elliott_wave_output

        self.volume_series = volume_series if volume_series is not None else []
        self.sentiment_signal = np.clip(sentiment_signal, -1.0, 1.0)
        self.macro_sync_state = np.clip(macro_sync_state, 0.0, 1.0)
        self.spiral_entropy = np.clip(spiral_entropy, 0.0, 1.0)
        self.mutation_level = np.clip(mutation_level, 0.0, 1.0)
        self.phase = phase

        logger.info(f"ImpulseNeuron initialized | phase={self.phase} | entropy={self.spiral_entropy:.3f} | mutation={self.mutation_level:.3f}")

    def _sanitize_impulse_waves(self, waves):
        sanitized = []
        for wave in waves:
            try:
                price = float(wave.get("price", 0))
            except Exception as e:
                logger.warning(f"ImpulseNeuron: Failed to convert price to float: {wave.get('price')} | {e}")
                price = 0.0
            try:
                time = float(wave.get("time", 0))
            except Exception as e:
                logger.warning(f"ImpulseNeuron: Failed to convert time to float: {wave.get('time')} | {e}")
                time = 0.0
            sanitized.append({"price": price, "time": time})
        return sanitized

    def compute(self):
        impulse_waves_raw = self.ew_output.get("impulse_waves", [])
        impulse_waves = self._sanitize_impulse_waves(impulse_waves_raw)

        wave_prob_score = self.ew_output.get("wave_probability_score", 0.0)
        wave_confidence = self.ew_output.get("wave_confidence", 0.0)

        if not impulse_waves or len(impulse_waves) < 3 or wave_confidence < 0.3:
            logger.warning("Low wave confidence or insufficient impulse waves; impulse strength set to 0")
            return {
                "impulse_score_norm": 0.0,
                "impulse_confidence": 0.0,
                "impulse_uncertainty": 0.0,
                "reflex_latency": 1.0,
                "impulse_recommendation": "weak_impulse",
                "volume_factor": 1.0,
                "sentiment_factor": 1.0,
                "macro_factor": 1.0,
                "phase_weight": 1.0,
                "base_impulse_strength": 0.0,
            }

        try:
            prices = np.array([w["price"] for w in impulse_waves], dtype=np.float64)
            times = np.array([w["time"] for w in impulse_waves], dtype=np.float64)
        except Exception as e:
            logger.error(f"Error converting impulse wave data to numpy arrays: {e}")
            return {
                "impulse_score_norm": 0.0,
                "impulse_confidence": 0.0,
                "impulse_uncertainty": 0.0,
                "reflex_latency": 1.0,
                "impulse_recommendation": "weak_impulse",
                "volume_factor": 1.0,
                "sentiment_factor": 1.0,
                "macro_factor": 1.0,
                "phase_weight": 1.0,
                "base_impulse_strength": 0.0,
            }

        lengths = np.abs(np.diff(prices))
        delta_ts = np.diff(times)
        # Prevent division by zero or negative time intervals
        delta_ts[delta_ts <= 0] = 1.0
        velocity = lengths / delta_ts

        avg_length = np.mean(lengths) if lengths.size else 0.0
        avg_velocity = np.mean(velocity) if velocity.size else 0.0
        impulse_variance = np.var(lengths) if lengths.size > 1 else 0.0

        base_impulse_strength = avg_length * avg_velocity * wave_prob_score * wave_confidence
        base_impulse_strength = np.clip(base_impulse_strength / 10.0, 0.0, 1.0)

        logger.debug(f"Base impulse | length={avg_length:.4f} velocity={avg_velocity:.4f} variance={impulse_variance:.6f} strength={base_impulse_strength:.4f}")

        volume_factor = self._compute_volume_factor()
        sentiment_factor = 1 + 0.2 * self.sentiment_signal
        macro_factor = self.macro_sync_state
        entropy_damping = 1 - self.spiral_entropy
        mutation_damping = 1 - self.mutation_level

        adaptive_multiplier = volume_factor * sentiment_factor * macro_factor * entropy_damping * mutation_damping
        adaptive_multiplier = np.clip(adaptive_multiplier, 0.0, 2.0)

        phase_weight = self._phase_weighting()
        adjusted_strength = base_impulse_strength * adaptive_multiplier * phase_weight
        adjusted_strength = np.clip(adjusted_strength, 0.0, 1.0)

        uncertainty = np.tanh(impulse_variance * 10)
        confidence = 1 - uncertainty
        reflex_latency = np.clip(1 - confidence, 0.0, 1.0)

        if adjusted_strength > 0.75 and confidence > 0.7:
            recommendation = "strong_buy" if self.phase in ["rebirth", "growth"] else "risk_caution"
        elif adjusted_strength < 0.3 or confidence < 0.4:
            recommendation = "weak_impulse"
        else:
            recommendation = "neutral"

        result = {
            "impulse_score_norm": round(adjusted_strength, 4),
            "impulse_confidence": round(confidence, 4),
            "impulse_uncertainty": round(uncertainty, 4),
            "reflex_latency": round(reflex_latency, 4),
            "impulse_recommendation": recommendation,
            "volume_factor": round(volume_factor, 4),
            "sentiment_factor": round(sentiment_factor, 4),
            "macro_factor": round(macro_factor, 4),
            "phase_weight": round(phase_weight, 4),
            "base_impulse_strength": round(base_impulse_strength, 4),
        }

        logger.info(f"ImpulseNeuron compute result: {result}")
        return result

    def _compute_volume_factor(self):
        if not self.volume_series or len(self.volume_series) == 0:
            return 1.0
        recent_vol = np.mean(self.volume_series[-10:])
        max_vol = np.max(self.volume_series[-50:]) if len(self.volume_series) >= 50 else recent_vol
        vol_factor = recent_vol / (max_vol + 1e-8)
        vol_factor = np.clip(vol_factor, 0.0, 1.0)
        logger.debug(f"Volume factor: {vol_factor:.4f}")
        return vol_factor

    def _phase_weighting(self):
        weights = {
            "rebirth": 1.4,
            "growth": 1.2,
            "neutral": 1.0,
            "decay": 0.7,
            "death": 0.5
        }
        weight = weights.get(self.phase, 1.0)
        logger.debug(f"Phase weighting: phase={self.phase} weight={weight}")
        return weight
