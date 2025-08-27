#!/usr/bin/env python3
# ================================================================
# 🧠 WEPS StateVectorBuilder — Institutional Spiral Intelligence Core
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Aggregates comprehensive neuron outputs into a fixed-length, normalized state vector
#   - Includes spiral phase encoding, epigenetic gene activity, and multi-timeframe indicators
#   - Engineered for maximal reliability, extensibility, and scientific rigor
# ================================================================

import numpy as np
import logging

# Assumes MultiTFIndicators is implemented in weps.utils.multi_tf_indicators
from weps.utils.multi_tf_indicators import MultiTFIndicators  

logger = logging.getLogger("WEPS.StateVectorBuilder")

class StateVectorBuilder:
    def __init__(self, neuron_outputs: dict, gene_map: dict, spiral_phase: str, spiral_z: float, multi_tf_dfs: dict):
        """
        :param neuron_outputs: Dict[str, dict] - neuron name -> output dict
        :param gene_map: Dict[str, float] - epigenetic gene activity values
        :param spiral_phase: str - current spiral phase ("rebirth", "growth", "decay", "death", "neutral")
        :param spiral_z: float - spiral wave confidence (0..1)
        :param multi_tf_dfs: Dict[str, pd.DataFrame] - OHLCV data keyed by timeframe, for multi-TF indicators
        """
        self.neuron_outputs = neuron_outputs or {}
        self.gene_map = gene_map or {}
        self.spiral_phase = spiral_phase or "neutral"
        self.spiral_z = self._clamp(spiral_z, 0.0, 1.0)
        self.multi_tf_dfs = multi_tf_dfs or {}

        # Stable ordered neuron keys for vector consistency
        self.expected_neurons = [
            "momentum", "impulse", "weakness", "volatility", "risk_defense",
            "cycle", "elliott_wave", "fibonacci", "candle_pattern",
            "immune_response", "trend", "liquidity", "sentiment",
            "correlation", "support_resistance", "half_life",
            "fractal", "memory_trauma", "metabolic", "macro_sync",
            "spiral_density"
        ]

    def build_state_vector(self) -> np.ndarray:
        features = []

        # Extract neuron scalars
        for neuron in self.expected_neurons:
            neuron_output = self.neuron_outputs.get(neuron, {})
            scalars = self._extract_scalars_for_neuron(neuron, neuron_output)
            features.extend(scalars)

        # Spiral phase one-hot encoding
        phase_encoding = self._encode_spiral_phase(self.spiral_phase)
        features.extend(phase_encoding)

        # Append spiral confidence score
        features.append(self.spiral_z)

        # Epigenetic gene vector normalized
        gene_vector = self._extract_gene_vector()
        features.extend(gene_vector)

        # Multi-timeframe technical indicators as normalized scalars
        multi_tf_features = self._extract_multi_tf_indicators()
        features.extend(multi_tf_features)

        # Final normalization and clipping
        vector = np.array(features, dtype=np.float32)
        normalized_vector = self._normalize_vector(vector)

        logger.info(f"StateVectorBuilder: Built state vector length={len(normalized_vector)} phase={self.spiral_phase}")
        return normalized_vector

    def _extract_scalars_for_neuron(self, neuron_key: str, output: dict) -> list:
        """
        Extract meaningful normalized scalars for a given neuron output dictionary.
        Returns a list of floats between 0 and 1.
        """
        try:
            if not isinstance(output, dict) or not output:
                return [0.0]

            scalars = []

            def safe_get(keys):
                for key in keys:
                    if key in output:
                        val = output[key]
                        if isinstance(val, (float, int)):
                            return self._clamp(float(val))
                return 0.0

            if neuron_key == "momentum":
                scalars.append(safe_get(["momentum_score_norm", "score", "value"]))
            elif neuron_key == "impulse":
                scalars.append(safe_get(["impulse_score_norm", "score", "value"]))
            elif neuron_key == "weakness":
                scalars.append(safe_get(["weakness_score_norm", "score", "value"]))
            elif neuron_key == "volatility":
                scalars.append(safe_get(["volatility_score_norm", "score", "value"]))
            elif neuron_key == "risk_defense":
                scalars.append(safe_get(["risk_defense_score", "score", "value"]))
            elif neuron_key == "cycle":
                scalars.append(safe_get(["cycle_amplitude"]))
                scalars.append(safe_get(["cycle_confidence"]))
            elif neuron_key == "elliott_wave":
                scalars.append(safe_get(["wave_confidence"]))
                scalars.append(1.0 if output.get("valid_impulse", False) else 0.0)
                wave_count = output.get("total_candidate_waves", 0)
                scalars.append(min(wave_count / 500.0, 1.0))
            elif neuron_key == "fibonacci":
                scalars.append(safe_get(["retrace_confidence", "score"]))
            elif neuron_key == "candle_pattern":
                scalars.append(safe_get(["pattern_score", "score"]))
            elif neuron_key == "immune_response":
                scalars.append(safe_get(["immunity_confidence"]))
                scalars.append(safe_get(["destruction_index"]))
            elif neuron_key == "trend":
                scalars.append(safe_get(["trend_strength", "score"]))
            elif neuron_key == "liquidity":
                scalars.append(safe_get(["liquidity_score", "score"]))
            elif neuron_key == "sentiment":
                scalars.append(safe_get(["sentiment_score", "score"]))
            elif neuron_key == "correlation":
                scalars.append(safe_get(["correlation_score", "score"]))
            elif neuron_key == "support_resistance":
                scalars.append(safe_get(["support_strength", "score"]))
            elif neuron_key == "half_life":
                scalars.append(safe_get(["half_life_norm", "score"]))
            elif neuron_key == "fractal":
                scalars.append(safe_get(["fractal_strength", "score"]))
            elif neuron_key == "memory_trauma":
                scalars.append(safe_get(["trauma_score", "score"]))
            elif neuron_key == "metabolic":
                scalars.append(safe_get(["metabolic_rate", "score"]))
            elif neuron_key == "macro_sync":
                scalars.append(safe_get(["macro_correlation", "score"]))
            elif neuron_key == "spiral_density":
                scalars.append(safe_get(["density_score", "score"]))
            else:
                # Generic fallback: first numeric value found
                for val in output.values():
                    if isinstance(val, (float, int)):
                        scalars.append(self._clamp(float(val)))
                        break

            if not scalars:
                scalars = [0.0]

            return [self._clamp(s) for s in scalars]

        except Exception as ex:
            logger.error(f"StateVectorBuilder: Error extracting scalars for neuron '{neuron_key}': {ex}")
            return [0.0]

    def _extract_gene_vector(self) -> list:
        genes = sorted(self.gene_map.keys())
        vector = []
        for gene in genes:
            val = self.gene_map.get(gene, 0.0)
            vector.append(self._clamp(float(val)))
        return vector

    def _encode_spiral_phase(self, phase: str) -> list:
        phases = ["rebirth", "growth", "decay", "death", "neutral"]
        return [1.0 if phase == p else 0.0 for p in phases]

    def _extract_multi_tf_indicators(self) -> list:
        features = []
        try:
            mtf = MultiTFIndicators(self.multi_tf_dfs)
            indicators = mtf.extract_all()
            for key in sorted(indicators.keys()):
                val = indicators[key]
                features.append(self._clamp(float(val)))
        except Exception as ex:
            logger.warning(f"StateVectorBuilder: Multi-TF indicators extraction failed: {ex}")
        return features

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        return np.clip(vector, 0.0, 1.0)

    def _clamp(self, val: float, min_val=0.0, max_val=1.0) -> float:
        if val is None or not isinstance(val, (float, int)):
            return 0.0
        return max(min_val, min(val, max_val))
