#!/usr/bin/env python3
# ====================================================================
# 🧬 WEPS LiveDataFeeder v6.5 — NVIDIA-Tier Spiral Intelligence Feeder
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Real-time OHLCV-to-Cortex payload engine for WEPS organisms
#   - Spiral lifecycle ready: DNA → Phase → Neurons → Entropy → Reflex
#   - Mutation-proof state vector validation and auto-healing
#   - Logs candle pattern names, entropy, SEI, spiral Z, and full neuron context
# ====================================================================

import logging
from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np

from weps.utils.live_data_updater import LiveDataUpdater
from weps.core.weps_pipeline_interface import WEPSPipelineInterface

logger = logging.getLogger("WEPS.LiveDataFeeder")
logger.setLevel(logging.INFO)


class LiveDataFeeder:
    def __init__(
        self,
        organism: str,
        timeframes: Optional[List[str]] = None,
        preferred_price_tf: str = "1d"
    ) -> None:
        self.organism = organism.upper()
        self.timeframes = timeframes or ["1h", "4h", "1d"]
        self.preferred_price_tf = preferred_price_tf if preferred_price_tf in self.timeframes else self.timeframes[0]

        self.live_updater = LiveDataUpdater()
        self.pipeline_interface = WEPSPipelineInterface(self.organism, self.timeframes)
        self.current_index = 0
        self.max_index = 0

        logger.info(f"[LiveDataFeeder] Initialized for {self.organism} | TFs: {self.timeframes}")

    def initialize(self, initial_data: Dict[str, pd.DataFrame]) -> None:
        if not initial_data:
            raise ValueError("No initial data provided.")

        for tf in self.timeframes:
            df = initial_data.get(tf)
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                raise ValueError(f"Invalid or missing DataFrame for {tf}")
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                raise TypeError(f"{tf} index must be datetime64")
            if not df.index.is_monotonic_increasing:
                raise ValueError(f"{tf} index must be sorted ascending")
            self.live_updater.dfs[tf] = df.copy()

        self.max_index = min(len(df) for df in self.live_updater.dfs.values()) - 1
        self.current_index = 0
        self._sync_pipeline_dfs()

        logger.info(f"[LiveDataFeeder] Initialization complete. Max index: {self.max_index}")

    def push_new_candles(self, new_data: Dict[str, pd.DataFrame]) -> None:
        if not new_data:
            raise ValueError("No new candle data provided.")

        for tf in self.timeframes:
            df_new = new_data.get(tf)
            if df_new is None or not isinstance(df_new, pd.DataFrame) or df_new.empty:
                continue
            if not pd.api.types.is_datetime64_any_dtype(df_new.index):
                raise TypeError(f"{tf} index must be datetime64")
            existing_df = self.live_updater.dfs.get(tf)
            if existing_df is not None:
                overlap = existing_df.index.intersection(df_new.index)
                df_new = df_new[~df_new.index.isin(overlap)]
                updated_df = pd.concat([existing_df, df_new]).sort_index()
            else:
                updated_df = df_new.copy()
            self.live_updater.dfs[tf] = updated_df

        self.max_index = min(len(df) for df in self.live_updater.dfs.values()) - 1
        self._sync_pipeline_dfs()
        logger.info(f"[LiveDataFeeder] New candles pushed. Max index: {self.max_index}")

    def _sync_pipeline_dfs(self) -> None:
        sliced_dfs = self.live_updater.get_data_slices(self.max_index)
        self.pipeline_interface.dfs = sliced_dfs
        self.pipeline_interface.pipeline.dfs = sliced_dfs

    def _sanitize_neuron_outputs(self, neuron_outputs: Optional[Dict[str, Any]]) -> Dict[str, Dict]:
        required_keys = [
            "impulse", "volatility", "trend", "sentiment",
            "weakness", "risk_defense", "candle_pattern"
        ]
        sanitized = {}
        if neuron_outputs is None or not isinstance(neuron_outputs, dict):
            logger.warning(f"[{self.organism}] Neuron outputs missing or malformed; applying safe defaults.")
            neuron_outputs = {}

        for key in required_keys:
            value = neuron_outputs.get(key)
            if not isinstance(value, dict):
                logger.warning(f"[{self.organism}] Neuron '{key}' missing or invalid; setting safe default.")
                sanitized[key] = {}
            else:
                sanitized[key] = value
        return sanitized

    def build_live_payload(self, symbol: str, fresh_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        🔬 WEPS Live Payload Constructor — Reflex Cortex Real-Time Bridge
        Converts real-time OHLCV into a biologically structured state vector.
        Logs spiral phase, entropy, SEI, Z-score, and candle pattern details.
        """
        self.pipeline_interface.dfs = fresh_dfs
        self.pipeline_interface.pipeline.dfs = fresh_dfs
        self.pipeline_interface.pipeline.run()

        pipeline = self.pipeline_interface.pipeline
        state_vector = pipeline.state_vector
        raw_outputs = pipeline.neuron_outputs
        wave_result = pipeline.spiral_wave_result or {}
        dna_vector = pipeline.dna_vector if pipeline.dna_vector is not None else np.zeros(1)

        # Auto-heal malformed numpy vector (legacy bug fallback)
        if isinstance(state_vector, np.ndarray):
            logger.warning(f"[{symbol}] 🩺 Auto-healing malformed state_vector (ndarray)...")
            neuron_keys = ["impulse", "volatility", "trend", "sentiment"]
            state_vector = {
                key: {"score": float(state_vector[i])}
                for i, key in enumerate(neuron_keys[:len(state_vector)])
            }

        neuron_outputs = self._sanitize_neuron_outputs(raw_outputs)

        # Validate structure
        if not isinstance(state_vector, dict):
            raise ValueError(f"[{symbol}] ❌ Invalid state_vector: expected dict, got {type(state_vector)}")
        required_keys = ["impulse", "volatility", "trend", "sentiment"]
        missing_keys = [k for k in required_keys if k not in state_vector or not isinstance(state_vector[k], dict)]
        if missing_keys:
            raise ValueError(f"[{symbol}] ❌ Missing required neuron outputs: {missing_keys}")

        # Extract candle pattern summary
        candle_data = neuron_outputs.get("candle_pattern", {}).get("summary", {})
        pattern_name = candle_data.get("detected_pattern", "none")
        pattern_class = candle_data.get("transition_type", "neutral")
        pattern_valid = True if pattern_name != "none" else False

        entropy = float(np.std(dna_vector))
        sei = float(np.mean(dna_vector))

        context = {
            "organism": self.organism,
            "phase": pipeline.final_spiral_phase,
            "entropy": entropy,
            "sei": sei,
            "sentiment": pipeline.sentiment_score,
            "wave_result": wave_result,
            "neurons": neuron_outputs,
            "gene_map": pipeline.gene_map or {},
            "z_score": wave_result.get("z_score", 0.0),
            "volatility": getattr(pipeline, "volatility_score", 0.0),
            "timing_score": getattr(pipeline, "corrective_wave_timing_factor", 1.0),
            "raw_df": fresh_dfs.get("1h", None),
            "pattern": {
                "name": pattern_name,
                "class": pattern_class,
                "valid": pattern_valid
            }
        }

        logger.info(
            f"[{symbol}] 🔎 Phase={context['phase']} | Entropy={entropy:.4f} | SEI={sei:.4f} | "
            f"Pattern=🕯️ {pattern_name.upper()} ({pattern_class}) | Z={context['z_score']:.2f}"
        )

        return {
            "state_vector": state_vector,
            "context": context
        }
