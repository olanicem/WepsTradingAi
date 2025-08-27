#!/usr/bin/env python3
# ==========================================================
# 🌐 WEPS MacroSyncNeuron — Final Production Version (Phase-Aware)
# ✅ Detects Macro Alignment with Global Indices, Commodities & FX
# ✅ Computes Macro Synchronization Score & Dominant Driver
# ✅ Outputs Spiral Phase-Aware Macro Bias for Reflex Cortex
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import numpy as np
import logging

logger = logging.getLogger("WEPS.Neurons.MacroSyncNeuron")

class MacroSyncNeuron:
    """
    WEPS MacroSyncNeuron
    - Measures macro synchronization of asset with global economic drivers.
    - Correlates asset price with key indices, commodities, and currencies.
    - Outputs macro bias signal, phase-aware.
    """
    def __init__(self, df: dict, phase: str = "neutral"):
        """
        Args:
            df (dict): Dictionary of DataFrames with keys like 'S&P500', 'DXY', etc.
            phase (str): Current spiral phase.
        """
        self.df = df
        self.phase = phase
        logger.info("MacroSyncNeuron initialized with phase=%s", self.phase)

    def compute(self) -> dict:
        if not isinstance(self.df, dict) or not self.df:
            raise ValueError("MacroSyncNeuron requires dictionary of macro DataFrames.")

        sync_scores = {}
        for key, macro_df in self.df.items():
            if macro_df.empty or "close" not in macro_df.columns:
                continue
            macro_returns = np.diff(np.log(macro_df['close'].values))
            asset_returns = np.diff(np.log(self.df['asset']['close'].values))
            min_len = min(len(macro_returns), len(asset_returns))
            if min_len < 50:
                continue
            corr = np.corrcoef(asset_returns[-min_len:], macro_returns[-min_len:])[0, 1]
            sync_scores[key] = corr

        if not sync_scores:
            return {"macro_sync_score_norm": 0.0, "dominant_macro_driver": None, "macro_bias": "neutral"}

        dominant_macro = max(sync_scores, key=lambda k: abs(sync_scores[k]))
        macro_sync = sync_scores[dominant_macro]
        phase_adjusted = round(np.clip(self._adjust_score_by_phase(macro_sync), -1, 1), 4)

        macro_bias = "risk-on" if phase_adjusted > 0.3 else "risk-off" if phase_adjusted < -0.3 else "neutral"
        result = {
            "macro_sync_score_norm": phase_adjusted,
            "dominant_macro_driver": dominant_macro,
            "macro_bias": macro_bias
        }
        logger.info("MacroSyncNeuron completed: %s", result)
        return result

    def _adjust_score_by_phase(self, score: float) -> float:
        phase_weights = {"rebirth": 1.2, "growth": 1.0, "decay": 0.8, "neutral": 1.0}
        return score * phase_weights.get(self.phase, 1.0)
