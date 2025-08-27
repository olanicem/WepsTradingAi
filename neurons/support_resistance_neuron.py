#!/usr/bin/env python3
# ==========================================================
# 🟢 WEPS SupportResistanceNeuron — Final Production Version
# ✅ Dynamically Identifies & Evaluates Key Support & Resistance Zones
# ✅ Measures Bounce Frequency & Proximity to Critical Levels
# ✅ Outputs Phase-Aware Reaction Probability
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("WEPS.Neurons.SupportResistanceNeuron")

class SupportResistanceNeuron:
    """
    WEPS SupportResistanceNeuron
    - Detects dynamic support & resistance levels over multiple timeframes.
    - Measures proximity of current price to key zones and expected reactivity.
    - Outputs phase-aware probability of reversal or breakout.
    """
    def __init__(self, df: pd.DataFrame, phase: str = "neutral"):
        self.df = df
        self.phase = phase
        logger.info("SupportResistanceNeuron initialized with phase=%s", self.phase)

    def compute(self, df: pd.DataFrame = None) -> dict:
        df = df or self.df
        if df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
            raise ValueError("SupportResistanceNeuron requires dataframe with 'high', 'low', and 'close' columns.")

        if len(df) < 200:
            raise ValueError("SupportResistanceNeuron requires at least 200 data points.")

        # Calculate recent swing levels
        last_highs = [df['high'].rolling(w).max().iloc[-1] for w in [20, 50, 200]]
        last_lows = [df['low'].rolling(w).min().iloc[-1] for w in [20, 50, 200]]
        current_price = df['close'].iloc[-1]

        nearest_support = max([low for low in last_lows if low < current_price], default=None)
        nearest_resistance = min([high for high in last_highs if high > current_price], default=None)

        # Compute bounce frequency by counting tests of levels
        tolerance = 0.002 * current_price  # 0.2% threshold
        bounces_support = ((np.abs(df['close'] - nearest_support) <= tolerance).sum()) if nearest_support else 0
        bounces_resistance = ((np.abs(df['close'] - nearest_resistance) <= tolerance).sum()) if nearest_resistance else 0

        # Reaction probability based on phase & bounce strength
        reaction_prob = self._compute_reaction_probability(bounces_support, bounces_resistance)

        result = {
            "nearest_support": round(nearest_support, 4) if nearest_support else None,
            "nearest_resistance": round(nearest_resistance, 4) if nearest_resistance else None,
            "support_bounces": int(bounces_support),
            "resistance_bounces": int(bounces_resistance),
            "phase_aware_reaction_prob": round(reaction_prob, 4)
        }
        logger.info("SupportResistanceNeuron completed: %s", result)
        return result

    def _compute_reaction_probability(self, bounces_support, bounces_resistance) -> float:
        total_bounces = bounces_support + bounces_resistance
        base_prob = np.tanh(total_bounces / 10)  # scale probability non-linearly
        phase_weights = {"rebirth": 1.2, "growth": 1.0, "decay": 0.7, "neutral": 1.0}
        return np.clip(base_prob * phase_weights.get(self.phase, 1.0), 0, 1)
