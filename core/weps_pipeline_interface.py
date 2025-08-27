#!/usr/bin/env python3
# ================================================================
# 🧬 WEPSPipelineInterface — Final Spiral Intelligence Interface
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Wraps WEPSPipeline for RL or Real-Time Spiral Execution
#   - Exposes spiral phase, z-score, fibs, entropy, candle pattern, DNA, neuron map
# ================================================================

import logging
from typing import Tuple, Dict, Any, Optional, List

from weps.core.weps_pipeline import WEPSPipeline
import numpy as np

logger = logging.getLogger("WEPS.PipelineInterface")

class WEPSPipelineInterface:
    SUPPORTED_TIMEFRAMES: List[str] = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

    def __init__(self, organism: str, timeframes: Optional[List[str]] = None):
        self.organism = organism.upper()
        self.timeframes = timeframes if timeframes else ["1h", "4h", "1d"]

        invalid = [tf for tf in self.timeframes if tf not in self.SUPPORTED_TIMEFRAMES]
        if invalid:
            logger.warning(f"[{self.organism}] Invalid timeframes: {invalid}. Defaulting.")
            self.timeframes = ["1h", "4h", "1d"]

        self.pipeline = WEPSPipeline(self.organism, timeframes=self.timeframes)
        self.current_step: int = 0
        self.max_steps: int = 0
        self.dfs: Optional[Dict[str, Any]] = None
        self.data_length: int = 0

    def reset(self, start_index: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        logger.info(f"[{self.organism}] 🔁 Resetting PipelineInterface at index {start_index}")
        try:
            self.pipeline.run()
        except Exception as e:
            logger.error(f"[{self.organism}] ❌ Pipeline failed on reset: {e}")
            raise e

        self.dfs = self.pipeline.dfs
        lengths = [len(df) for df in self.dfs.values()]
        self.data_length = min(lengths)
        self.max_steps = self.data_length - 1

        if start_index >= self.max_steps:
            raise ValueError(f"Start index {start_index} >= max_steps {self.max_steps}")

        self.current_step = start_index
        return self._get_state_vector_and_info(self.current_step)

    def step(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.current_step >= self.max_steps:
            raise RuntimeError(f"[{self.organism}] Step exceeds max. Call reset.")

        self.current_step += 1
        self._update_dfs_to_current_step()

        try:
            self.pipeline.run()
        except Exception as e:
            logger.error(f"[{self.organism}] ❌ Pipeline run failed on step: {e}")
            raise e

        state_vector = self.pipeline.state_vector
        done = self.current_step >= self.max_steps

        info = {
            "organism": self.organism,
            "step": self.current_step,
            "done": done,
            "phase": self.pipeline.final_spiral_phase,
            "z_score": self.pipeline.spiral_wave_result.get("z_score", 0.0),
            "fib_levels": self.pipeline.spiral_wave_result.get("fib_levels", {}),
            "entropy": self.pipeline.spiral_wave_result.get("entropy", 0.0),
            "half_life": self.pipeline.spiral_wave_result.get("half_life", 0.0),
            "growth_score": self.pipeline.spiral_wave_result.get("growth_score", 0.0),
            "sentiment_score": self.pipeline.sentiment_score,
            "neurons": self.pipeline.neuron_outputs,  # 🧠 Includes CandlePatternNeuron v8
            "dna_vector": self.pipeline.dna_vector.tolist() if hasattr(self.pipeline, "dna_vector") else [],
            "gene_map": self.pipeline.gene_map,
        }

        return state_vector, info

    def _update_dfs_to_current_step(self) -> None:
        for tf in self.timeframes:
            df = self.dfs.get(tf)
            if df is None or df.empty:
                raise RuntimeError(f"Missing or empty dataframe for TF: {tf}")
            self.pipeline.dfs[tf] = df.iloc[: self.current_step + 1].copy()

    def _get_state_vector_and_info(self, step_index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        for tf in self.timeframes:
            df = self.pipeline.dfs.get(tf)
            if df is not None and not df.empty:
                self.pipeline.dfs[tf] = df.iloc[: step_index + 1].copy()

        try:
            self.pipeline.run()
        except Exception as e:
            logger.error(f"[{self.organism}] ❌ Pipeline run failed during reset fetch: {e}")
            raise e

        info = {
            "organism": self.organism,
            "step": step_index,
            "done": False,
            "phase": self.pipeline.final_spiral_phase,
            "z_score": self.pipeline.spiral_wave_result.get("z_score", 0.0),
            "fib_levels": self.pipeline.spiral_wave_result.get("fib_levels", {}),
            "entropy": self.pipeline.spiral_wave_result.get("entropy", 0.0),
            "half_life": self.pipeline.spiral_wave_result.get("half_life", 0.0),
            "growth_score": self.pipeline.spiral_wave_result.get("growth_score", 0.0),
            "sentiment_score": self.pipeline.sentiment_score,
            "neurons": self.pipeline.neuron_outputs,
            "dna_vector": self.pipeline.dna_vector.tolist() if hasattr(self.pipeline, "dna_vector") else [],
            "gene_map": self.pipeline.gene_map,
        }

        return self.pipeline.state_vector, info
