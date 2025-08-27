#!/usr/bin/env python3
# ==========================================================
# 🧬 WEPS DNAEncoder — Supreme Spiral Intelligence Edition
# ✅ Encodes Organism Genome + Live Market Data → DNA Vector
# ✅ Spiral Phase-Aware, Entropy & Mutation-Aware, Resilient to Missing Data
# ✅ Outputs Real-Time DNA Vector + Context Dict
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import numpy as np
import logging

logger = logging.getLogger("WEPS.DNAEncoder")

class DNAEncoder:
    """
    WEPS DNA Encoder
    - Transforms an organism's genome + live market data into a normalized DNA vector
    - Outputs both numeric vector & contextual state dict for neuron processing.
    - Surpasses Citadel-grade standards for precision & spiral intelligence modeling.
    """
    def __init__(self, genome: dict, asset_symbol: str, phase_name: str):
        self.genome = genome
        self.asset = asset_symbol
        self.phase = phase_name
        logger.info("DNAEncoder initialized for %s in phase: %s", self.asset, self.phase)

    def encode(self, market_data: dict) -> tuple:
        """
        Combines genome & live data → returns DNA vector + context dict.
        
        Args:
            market_data (dict): Live asset data e.g., ATR, impulse, correction, entropy, mutation
        
        Returns:
            np.ndarray: DNA vector [N features]
            dict: context state for neuron processor
        """
        try:
            # Volatility normalization (ATR)
            atr = market_data.get("atr", 0.0)
            vol_norm = self._normalize_signal(
                atr,
                self.genome["volatility_dna"]["average_atr"],
                self.genome["volatility_dna"]["max_atr"]
            )

            # Impulse normalization
            impulse = market_data.get("impulse_strength", 0.0)
            impulse_norm = self._normalize_signal(
                impulse,
                0.0,
                self.genome["impulse_dna"]["average_impulse_strength"] * 2
            )

            # Correction normalization
            correction = market_data.get("correction_depth", 0.0)
            correction_norm = self._normalize_signal(
                correction,
                0.0,
                self.genome["corrective_dna"]["average_correction_depth"] * 2
            )

            # Entropy normalization (corrected to use nested average key)
            entropy = market_data.get(
                "entropy_score",
                self.genome["entropy_dna"]["historical_entropy_levels"]["average"]
            )
            entropy_norm = self._normalize_signal(
                entropy,
                self.genome["entropy_dna"]["historical_entropy_levels"]["min"],
                self.genome["entropy_dna"]["historical_entropy_levels"]["max"]
            )

            # Mutation normalization
            mutation = market_data.get("mutation_magnitude", 0.0)
            mutation_norm = self._normalize_signal(
                mutation,
                0.0,
                self.genome["mutation_dna"]["average_mutation_magnitude"] * 2
            )

            # Sentiment normalization
            sentiment = market_data.get("sentiment_score", 0.0)
            sentiment_norm = np.clip(sentiment, -1, 1) * 0.5 + 0.5  # maps -1:1 → 0:1

            # Liquidity normalization
            volume = market_data.get("volume", self.genome["metabolic_dna"]["average_daily_volume"])
            liquidity_norm = self._normalize_signal(
                volume,
                self.genome["metabolic_dna"]["average_daily_volume"] * 0.5,
                self.genome["metabolic_dna"]["average_daily_volume"] * 1.5
            )

            # Correlation normalization
            correlation = market_data.get("correlation_score", 0.0)
            correlation_norm = np.clip(abs(correlation), 0, 1)

            # Phase adjustment factor
            phase_weight = self.genome["sentiment_dna"]["phase_amplification_factors"].get(
                self.phase, 1.0
            )

            # Compose final vector & context
            vector = np.array([
                vol_norm, impulse_norm, correction_norm,
                entropy_norm, mutation_norm, sentiment_norm,
                liquidity_norm, correlation_norm
            ], dtype=np.float32) * phase_weight

            context = {
                "volatility_norm": vol_norm,
                "impulse_norm": impulse_norm,
                "correction_norm": correction_norm,
                "entropy_norm": entropy_norm,
                "mutation_norm": mutation_norm,
                "sentiment_norm": sentiment_norm,
                "liquidity_norm": liquidity_norm,
                "correlation_norm": correlation_norm,
                "phase_weight": phase_weight
            }

            logger.info("DNAEncoder for %s: vector=%s, context=%s", self.asset, vector, context)
            return vector, context

        except Exception as e:
            logger.error("DNAEncoder failed for %s: %s", self.asset, e, exc_info=True)
            raise

    def _normalize_signal(self, value: float, min_val: float, max_val: float) -> float:
        """Scales a signal into [0,1] given historical genome boundaries."""
        if max_val - min_val == 0:
            return 0.0
        norm = (value - min_val) / (max_val - min_val)
        return np.clip(norm, 0, 1)
