#!/usr/bin/env python3
# ==========================================================
# ⏰ WEPS Spiral BioclockNeuron v3.0 — FESI Compliant
# ✅ Market Rhythmic Alignment + Spiral Life Timing Engine
# ✅ Adaptive Phase-Aware Time Decay + Corrective Normalization
# ✅ Fourier Session Mapping + Bio-Heartbeat Scoring
# Author: Ola Bode (WEPS Creator)
# ==========================================================

import pandas as pd
import numpy as np
import logging
from datetime import datetime, time

logger = logging.getLogger("WEPS.Neurons.BioclockNeuron")

class BioclockNeuron:
    def __init__(self,
                 df: pd.DataFrame,
                 phase_name: str = "neutral",
                 t_phase_start: datetime = None,
                 t_corrective_start: datetime = None,
                 t_impulse_prior_duration: float = None,
                 t_expected_phase_avg: float = None) -> None:
        """
        Args:
            df (pd.DataFrame): OHLCV with datetime index or 'date' column.
            phase_name (str): Current spiral phase (rebirth, growth, decay, etc).
            t_phase_start (datetime): When current phase began.
            t_corrective_start (datetime): Start time of corrective wave.
            t_impulse_prior_duration (float): Duration of last impulse wave.
            t_expected_phase_avg (float): Avg duration for this phase in seconds.
        """
        self.df = df.copy()
        self.phase = phase_name
        self.t_phase_start = t_phase_start
        self.t_corrective_start = t_corrective_start
        self.t_impulse_prior_duration = t_impulse_prior_duration
        self.t_expected_phase_avg = t_expected_phase_avg
        logger.info(f"[BioclockNeuron] Initialized: phase={phase_name}")

    def compute(self, df: pd.DataFrame = None) -> dict:
        df = df or self.df
        if "date" not in df.columns:
            raise ValueError("DataFrame must contain 'date' column with datetime values.")
        df["date"] = pd.to_datetime(df["date"])
        t_now = df["date"].iloc[-1].to_pydatetime()

        # 1️⃣ Phase Elapsed Time Normalization
        x_phase = 0.0
        if self.t_phase_start and self.t_expected_phase_avg:
            elapsed = (t_now - self.t_phase_start).total_seconds()
            x_phase = elapsed / (self.t_expected_phase_avg + 1e-9)

        # 2️⃣ Corrective Elapsed Time vs Expected (based on golden 0.63 ratio)
        x_corrective = 0.0
        if self.t_corrective_start and self.t_impulse_prior_duration:
            t_elapsed = (t_now - self.t_corrective_start).total_seconds()
            t_expected = 0.63 * self.t_impulse_prior_duration
            x_corrective = t_elapsed / (t_expected + 1e-9)

        # 3️⃣ Phase-Weighted Biological Adjustment
        base_factor = min(x_phase, x_corrective)
        phase_adj = self._adjust_by_phase(base_factor)

        # 4️⃣ Market Rhythm Alignment Score (London/NY/Asia)
        session_alignment = self._compute_session_alignment(t_now)

        # 5️⃣ Final Composite Score
        result = {
            "timing_factor": round(x_corrective, 4),
            "phase_elapsed_factor": round(x_phase, 4),
            "phase_adjusted_timing": round(phase_adj, 4),
            "rhythm_alignment": session_alignment["session_name"],
            "peak_hours_score": round(session_alignment["peak_score"], 4)
        }
        logger.info("[BioclockNeuron] Computed: %s", result)
        return result

    def _adjust_by_phase(self, base: float) -> float:
        """
        Applies bio-weighted adjustment to timing based on current spiral phase.
        """
        weights = {
            "rebirth": 1.25,   # fast, impulsive
            "growth": 1.0,     # stable
            "decay": 0.75,     # sluggish, prolonged
            "death": 0.5,      # minimal reactivity
            "neutral": 1.0
        }
        multiplier = weights.get(self.phase, 1.0)
        return np.clip(base * multiplier, 0, 2)

    def _compute_session_alignment(self, current_time: datetime) -> dict:
        """
        Returns the current trading session and alignment score (peak=1.0, low=0.0).
        """
        hour = current_time.hour
        if 7 <= hour < 10:
            return {"session_name": "LondonOpen", "peak_score": 1.0}
        elif 13 <= hour < 16:
            return {"session_name": "NewYorkOpen", "peak_score": 0.9}
        elif 0 <= hour < 4:
            return {"session_name": "AsiaOpen", "peak_score": 0.8}
        elif 16 <= hour < 18:
            return {"session_name": "NY-LondonOverlap", "peak_score": 1.2}
        else:
            return {"session_name": "OffPeak", "peak_score": 0.3}
