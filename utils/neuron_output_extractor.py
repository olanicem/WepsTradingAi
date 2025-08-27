#!/usr/bin/env python3
# ==============================================================
# 🧠 WEPS Neuron Output Extractor — Institutional Spiral Intelligence
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Extracts and normalizes outputs from all WEPS neuron modules
#   - Prepares a unified flat dictionary of numeric features for state vector
#   - Applies fallback defaults and clamps values to [0,1]
#   - Engineered for traceability, extendability, and institutional rigor
# ==============================================================

import numpy as np
import logging

logger = logging.getLogger("WEPS.NeuronOutputExtractor")

class NeuronOutputExtractor:
    def __init__(self, neuron_outputs: dict):
        """
        neuron_outputs: dict
            Keys: neuron module names (str)
            Values: neuron output dicts (dict)
        """
        self.neuron_outputs = neuron_outputs

    def extract_features(self) -> dict:
        features = {}

        # Helper function to safely extract and normalize floats
        def safe_norm(value, default=0.0, vmin=0.0, vmax=1.0):
            try:
                val = float(value)
                norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else val
                return max(0.0, min(norm_val, 1.0))
            except Exception:
                return default

        # momentum_neuron
        momentum = self.neuron_outputs.get("momentum", {})
        features["momentum_score"] = safe_norm(momentum.get("momentum_score_norm", 0.0))

        # impulse_neuron
        impulse = self.neuron_outputs.get("impulse", {})
        features["impulse_score"] = safe_norm(impulse.get("impulse_score_norm", 0.0))
        # Encode impulse recommendation as categorical numeric: strong=1.0, weak=0.0, neutral=0.5
        impulse_rec_map = {"strong_impulse": 1.0, "weak_impulse": 0.0, "neutral": 0.5}
        rec = impulse.get("impulse_recommendation", "neutral")
        features["impulse_recommendation"] = impulse_rec_map.get(rec, 0.5)

        # weakness_neuron
        weakness = self.neuron_outputs.get("weakness", {})
        features["weakness_score"] = safe_norm(weakness.get("weakness_score", 0.0))

        # volatility_neuron
        volatility = self.neuron_outputs.get("volatility", {})
        features["volatility_score"] = safe_norm(volatility.get("volatility_score_norm", 0.0))
        features["atr"] = safe_norm(volatility.get("atr", 0.0), vmin=0, vmax=0.1)  # example scale

        # immune_response_neuron
        immune = self.neuron_outputs.get("immune_response", {})
        features["immune_confidence"] = safe_norm(immune.get("immunity_confidence", 0.0))
        features["early_exit_signal"] = 1.0 if immune.get("early_exit_signal", False) else 0.0
        features["destruction_index"] = safe_norm(immune.get("destruction_index", 0.0))

        # momentum_neuron - alternative momentum score if present
        features["momentum_score_alt"] = safe_norm(momentum.get("momentum_score_alt", 0.0))

        # bioclock_neuron
        bioclock = self.neuron_outputs.get("bioclock", {})
        features["bioclock_phase"] = safe_norm(bioclock.get("phase_score", 0.0))

        # candle_pattern_neuron
        candle = self.neuron_outputs.get("candle_pattern", {})
        pattern_score = safe_norm(candle.get("pattern_score", 0.0))
        features["candle_pattern_score"] = pattern_score
        # Categorical encoding for detected pattern (simplified, extend as needed)
        pattern = candle.get("detected_pattern", "neutral")
        known_patterns = {"tweezer_top": 1.0, "tweezer_bottom": 0.0, "neutral": 0.5}
        features["candle_pattern_detected"] = known_patterns.get(pattern, 0.5)

        # correlation_neuron
        correlation = self.neuron_outputs.get("correlation", {})
        features["correlation_score"] = safe_norm(correlation.get("correlation_coefficient", 0.0), vmin=-1.0, vmax=1.0)

        # fractal_neuron
        fractal = self.neuron_outputs.get("fractal", {})
        features["fractal_dimension"] = safe_norm(fractal.get("fractal_dimension", 0.0))
        features["hurst_exponent"] = safe_norm(fractal.get("hurst_exponent", 0.5))

        # half_life_neuron
        half_life = self.neuron_outputs.get("half_life", {})
        features["half_life_position"] = safe_norm(half_life.get("position_norm", 0.0))

        # liquidity_neuron
        liquidity = self.neuron_outputs.get("liquidity", {})
        features["liquidity_score"] = safe_norm(liquidity.get("liquidity_score_norm", 0.0))

        # macro_sync_neuron
        macro = self.neuron_outputs.get("macro_sync", {})
        features["macro_sync_score"] = safe_norm(macro.get("sync_score", 0.0))

        # memory_trauma_neuron
        trauma = self.neuron_outputs.get("memory_trauma", {})
        features["memory_trauma_score"] = safe_norm(trauma.get("trauma_score", 0.0))

        # metabolic_neuron
        metabolic = self.neuron_outputs.get("metabolic", {})
        features["metabolic_rate"] = safe_norm(metabolic.get("metabolic_rate_norm", 0.0))

        # phase_transition_neuron
        phase_transition = self.neuron_outputs.get("phase_transition", {})
        features["phase_transition_prob"] = safe_norm(phase_transition.get("transition_probability", 0.0))

        # reflex_neuron
        reflex = self.neuron_outputs.get("reflex", {})
        features["reflex_activation"] = safe_norm(reflex.get("activation_level", 0.0))

        # sentiment_neuron
        sentiment = self.neuron_outputs.get("sentiment", {})
        features["sentiment_score"] = safe_norm(sentiment.get("sentiment_adjustment", 0.0))

        # spiral_density_neuron
        spiral_density = self.neuron_outputs.get("spiral_density", {})
        features["spiral_density"] = safe_norm(spiral_density.get("density_norm", 0.0))

        # support_resistance_neuron
        sr = self.neuron_outputs.get("support_resistance", {})
        features["support_level_norm"] = safe_norm(sr.get("support_level_norm", 0.0))
        features["resistance_level_norm"] = safe_norm(sr.get("resistance_level_norm", 0.0))

        # trend_neuron
        trend = self.neuron_outputs.get("trend", {})
        features["trend_strength"] = safe_norm(trend.get("trend_strength_norm", 0.0))

        logger.info(f"Extracted {len(features)} neuron features for state vector.")
        return features


# Example usage:
# extractor = NeuronOutputExtractor(neuron_outputs)
# state_vector_features = extractor.extract_features()
# Then convert to np.array for ML or RL input.
