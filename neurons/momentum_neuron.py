#!/usr/bin/env python3
# ==========================================================
# 🚀 WEPS MomentumNeuron — FESI-Compliant Spiral Intelligence Core
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Multi-horizon return velocity and acceleration with trend confirmation
#   - Phase-depth and entropy/mutation confidence modulation
#   - Volume and multi-timeframe momentum context integration
#   - Biological metaphors: metabolic rate, reflex latency, adaptation capacity
# ==========================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("WEPS.Neurons.MomentumNeuron")

class MomentumNeuron:
    def __init__(self, df: pd.DataFrame, phase: str = "neutral", phase_depth: float = 0.0,
                 spiral_entropy: float = 0.5, mutation_level: float = 0.0,
                 volume_series: np.ndarray = None, multi_tf_momentum: dict = None):
        """
        :param df: OHLCV DataFrame with 'close' column, min 200 data points recommended
        :param phase: Spiral phase (rebirth/growth/decay/neutral/death)
        :param phase_depth: normalized depth in current phase (0..1)
        :param spiral_entropy: current entropy [0..1]
        :param mutation_level: mutation level [0..1]
        :param volume_series: recent volume array for liquidity modulation
        :param multi_tf_momentum: dict of multi-TF momentum scores normalized [0..1]
        """
        self.df = df
        self.phase = phase
        self.phase_depth = np.clip(phase_depth, 0.0, 1.0)
        self.spiral_entropy = np.clip(spiral_entropy, 0.0, 1.0)
        self.mutation_level = np.clip(mutation_level, 0.0, 1.0)
        self.volume_series = volume_series
        self.multi_tf_momentum = multi_tf_momentum or {}

        logger.info(f"MomentumNeuron initialized | phase={self.phase} depth={self.phase_depth:.3f} entropy={self.spiral_entropy:.3f} mutation={self.mutation_level:.3f}")

    def compute(self) -> dict:
        if self.df.empty or "close" not in self.df.columns:
            raise ValueError("MomentumNeuron requires dataframe with 'close' column.")
        closes = self.df['close'].values
        if len(closes) < 200:
            raise ValueError("MomentumNeuron requires at least 200 data points for robust analysis.")

        # Compute returns, velocity and acceleration
        returns = np.diff(closes) / closes[:-1]
        velocity = self._rolling_mean(returns, window=10)
        acceleration = self._rolling_mean(np.diff(velocity), window=5) if len(velocity) > 5 else np.array([0])

        velocity_score = np.clip(np.tanh(velocity[-1] * 10), -1, 1)
        acceleration_score = np.clip(np.tanh(acceleration[-1] * 20), -1, 1) if len(acceleration) > 0 else 0.0

        # Aggregate multi-timeframe momentum if present
        mtf_momentum_avg = np.mean(list(self.multi_tf_momentum.values())) if self.multi_tf_momentum else 0.5

        # Volume-based liquidity factor
        volume_factor = self._compute_volume_factor()

        # Combine scores with weights
        base_score = (0.5 * velocity_score) + (0.3 * acceleration_score) + (0.2 * (mtf_momentum_avg * 2 - 1))
        base_score = np.clip(base_score, -1, 1)

        # Phase depth modulation: momentum stronger deeper in growth/rebirth, weaker in decay
        depth_factor = self.phase_depth if self.phase in ["rebirth", "growth"] else (1 - self.phase_depth)

        # Entropy & mutation dampening uncertainty
        uncertainty_factor = (1 - self.spiral_entropy) * (1 - self.mutation_level)

        # Final adaptive momentum score
        adaptive_score = base_score * depth_factor * uncertainty_factor * volume_factor
        adaptive_score = np.clip(adaptive_score, -1, 1)

        # Normalize to [0,1]
        norm_score = (adaptive_score + 1) / 2

        # Confidence metric derived from phase depth and uncertainty
        confidence = np.clip(depth_factor * uncertainty_factor, 0, 1)

        # Reflex latency metaphor
        reflex_latency = 1 - confidence

        result = {
            "momentum_score_norm": round(norm_score, 4),
            "velocity_score": round(velocity_score, 4),
            "acceleration_score": round(acceleration_score, 4),
            "multi_tf_momentum": round(mtf_momentum_avg, 4),
            "volume_factor": round(volume_factor, 4),
            "phase_depth": round(self.phase_depth, 4),
            "entropy": round(self.spiral_entropy, 4),
            "mutation_level": round(self.mutation_level, 4),
            "confidence": round(confidence, 4),
            "reflex_latency": round(reflex_latency, 4),
        }

        logger.info(f"MomentumNeuron computed: {result}")
        return result

    def _rolling_mean(self, array: np.ndarray, window: int) -> np.ndarray:
        if len(array) < window:
            return array
        return np.convolve(array, np.ones(window)/window, mode='valid')

    def _compute_volume_factor(self) -> float:
        if self.volume_series is None or len(self.volume_series) == 0:
            return 1.0
        recent_vol = np.mean(self.volume_series[-10:])
        max_vol = np.max(self.volume_series[-50:]) if len(self.volume_series) >= 50 else recent_vol
        vol_factor = recent_vol / (max_vol + 1e-8)
        vol_factor = np.clip(vol_factor, 0, 1)
        logger.debug(f"Volume factor computed: {vol_factor:.4f}")
        return vol_factor
