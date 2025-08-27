#!/usr/bin/env python3
# ==========================================================
# 🔗 WEPS CorrelationNeuron v2.0 — Institutional Spiral Intelligence
# ✅ Analyzes Short & Medium-Term Correlation to Sister Asset
# ✅ Performs Granger Causality to Detect Predictive Power
# ✅ Outputs Phase-Aware Correlation Confidence
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.stattools import grangercausalitytests

logger = logging.getLogger("WEPS.Neurons.CorrelationNeuronV2")

class CorrelationNeuron:
    """
    WEPS Spiral CorrelationNeuron v2.0
    - Calculates real-time correlation with sister asset.
    - Confirms predictive relationships using Granger causality.
    - Outputs phase-aware correlation confidence.
    """
    def __init__(self, df_main: pd.DataFrame, df_sister: pd.DataFrame, phase: str = "neutral"):
        self.df_main = df_main
        self.df_sister = df_sister
        self.phase = phase
        logger.info("CorrelationNeuron initialized with phase=%s", self.phase)

    def compute(self, df_main: pd.DataFrame = None, df_sister: pd.DataFrame = None) -> dict:
        df_main = df_main or self.df_main
        df_sister = df_sister or self.df_sister
        if df_main.empty or df_sister.empty or 'close' not in df_main.columns or 'close' not in df_sister.columns:
            raise ValueError("CorrelationNeuron requires both dataframes with 'close' columns.")

        window_short, window_long = 10, 50
        if len(df_main) < window_long or len(df_sister) < window_long:
            raise ValueError("Insufficient data for correlation analysis (need ≥50 candles).")

        main_short = df_main['close'].iloc[-window_short:]
        sister_short = df_sister['close'].iloc[-window_short:]
        main_long = df_main['close'].iloc[-window_long:]
        sister_long = df_sister['close'].iloc[-window_long:]

        corr_short = np.corrcoef(main_short, sister_short)[0, 1]
        corr_long = np.corrcoef(main_long, sister_long)[0, 1]

        try:
            granger_result = grangercausalitytests(
                pd.concat([main_short.reset_index(drop=True), sister_short.reset_index(drop=True)], axis=1),
                maxlag=1, verbose=False
            )
            granger_pval = granger_result[1][0]['ssr_ftest'][1]
        except Exception as e:
            logger.warning("Granger causality failed: %s", e)
            granger_pval = 1.0

        corr_combined = 0.6 * corr_short + 0.4 * corr_long
        phase_adjusted = round(np.clip(self._adjust_confidence(corr_combined), -1, 1), 4)

        result = {
            "rolling_correlation": round(corr_combined, 4),
            "granger_pvalue": round(granger_pval, 4),
            "correlation_confidence": phase_adjusted
        }
        logger.info("CorrelationNeuron completed: %s", result)
        return result

    def _adjust_confidence(self, corr: float) -> float:
        phase_weights = {"rebirth": 1.2, "growth": 1.0, "decay": 0.8, "neutral": 1.0}
        adjusted = corr * phase_weights.get(self.phase, 1.0)
        logger.debug("Phase-adjusted correlation confidence: base=%.4f, phase=%s, adjusted=%.4f",
                     corr, self.phase, adjusted)
        return adjusted
