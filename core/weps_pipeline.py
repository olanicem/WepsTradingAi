#!/usr/bin/env python3
# =========================================================================
# 🧬 WEPSPipeline — Biological Spiral Intelligence Engine (Final Edition)
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Full spiral lifecycle analysis pipeline
#   - Genome loading → DNA encoding → Epigenetic gate → Neuron firing
#   - Final spiral phase judgment and state vector creation
# =========================================================================

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional

from weps.data_loader import WEPSMasterDataLoader
from weps.genome.genome_hub import load_genome
from weps.dna.dna_encoder import DNAEncoder
from weps.epigenetics.epigenetic_gate import EpigeneticGate
from weps.core.spiral_phase_detector import finalize_spiral_phase
from weps.core.state_vector_builder import StateVectorBuilder
from weps.utils.multi_tf_indicators import MultiTFIndicators
from weps.wave_engine.spiral_wave_engine import SpiralWaveEngineV2 as SpiralWaveEngine
from weps.neurons.neuron_processor import NeuronProcessor
from weps.neurons.sentiment_neuron import SentimentNeuron
from weps.neurons.candle_pattern_neuron import CandlePatternNeuron
from weps.neurons.trend_neuron import TrendNeuron
from weps.neurons.cycle_neuron import CycleNeuron
from weps.neurons.bioclock_neuron import BioclockNeuron

logger = logging.getLogger("WEPS.Pipeline")


class WEPSPipeline:
    def __init__(self, organism: str, timeframes: Optional[list] = None):
        self.organism = organism.upper()
        self.timeframes = timeframes or ["1h", "4h", "1d"]
        self.dfs: Dict[str, pd.DataFrame] = {}

        # Intelligence states
        self.genome = None
        self.dna_vector = None
        self.gene_map = None
        self.neuron_outputs = None
        self.spiral_wave_result = None
        self.final_spiral_phase = None
        self.state_vector = None

        # Diagnostics
        self.sentiment_score = 0.0
        self.candle_pattern_output = {}
        self.trend_output = {}
        self.multi_tf_corr_features = {}
        self.multi_tf_indicator_features = {}

    def ingest_live_ohlcv_dfs(self, dfs: Dict[str, pd.DataFrame]):
        for tf in self.timeframes:
            df = dfs.get(tf)
            if df is None or df.empty:
                raise ValueError(f"❌ Missing or empty OHLCV data for {tf}")
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                raise TypeError(f"❌ OHLCV index must be datetime for {tf}")
            self.dfs[tf] = df.copy()
        logger.info(f"[WEPSPipeline] ✅ Ingested live OHLCV for {self.organism}")

    def run(self):
        logger.info(f"🧬 [PIPELINE] Starting WEPSPipeline for {self.organism}...")

        # 1️⃣ Load data if not pre-ingested
        if not self.dfs:
            loader = WEPSMasterDataLoader()
            raw = loader.fetch_organisms([self.organism], limit=500)[self.organism]
            for tf in self.timeframes:
                df = raw["ohlcv_multi"].get(tf)
                if df is None or df.empty:
                    raise ValueError(f"❌ No data for {self.organism} @ {tf}")
                self.dfs[tf] = df.copy()

        # 2️⃣ Multi-Timeframe Feature Extraction
        self.multi_tf_corr_features = self.compute_multi_tf_correlation(self.dfs)
        mtf = MultiTFIndicators(self.dfs)
        self.multi_tf_indicator_features = mtf.extract_all()
        logger.info("📊 [MULTI-TF] Indicators and correlations extracted.")

        # 3️⃣ Spiral Wave Detection
        main_tf = self.timeframes[-1]
        window_df = self.dfs[main_tf].copy()
        wave_engine = SpiralWaveEngine(self.organism, window_df)
        self.spiral_wave_result = wave_engine.detect()
        wave_phase = self.spiral_wave_result.get("phase", "unknown")
        z_score = self.spiral_wave_result.get("z_score", 0.0)
        logger.info(f"🌊 [WAVE] Phase={wave_phase} | Z-Score={z_score:.4f}")

        # 4️⃣ Genome Loading & DNA Encoding
        self.genome = load_genome(self.organism)
        encoder = DNAEncoder(window_df, self.genome, extra_features=self.multi_tf_corr_features)
        self.dna_vector = encoder.encode()
        entropy = float(np.std(self.dna_vector))
        sei = float(np.mean(self.dna_vector))
        logger.info(f"🧠 [DNA] Encoded | Shape={self.dna_vector.shape} | SEI={sei:.4f} | Entropy={entropy:.4f}")

        # 5️⃣ Sentiment Neuron
        sentiment = SentimentNeuron(self.organism)
        sentiment_output = sentiment.compute(df=window_df, current_phase=wave_phase)
        self.sentiment_score = sentiment_output.get("sentiment_adjustment", 0.0)

        # 6️⃣ Candle Pattern Neuron
        candle = CandlePatternNeuron(phase=wave_phase)
        self.candle_pattern_output = candle.compute(df=window_df, current_phase=wave_phase)

        # 7️⃣ Trend Neuron
        trend = TrendNeuron(df=window_df, phase=wave_phase)
        self.trend_output = trend.compute()
        trend_state = self.trend_output.get("trend_state", "unknown")
        trend_strength = self.trend_output.get("trend_strength_score", 0.0)
        logger.info(f"📈 [TREND] State={trend_state} | Strength={trend_strength:.4f}")

        # 8️⃣ Cycle Neuron
        cycle = CycleNeuron(df=window_df, phase=wave_phase)
        cycle_output = cycle.compute()
        cycle_alignment = cycle_output.get("cycle_alignment", "unknown")
        cycle_strength = cycle_output.get("cycle_strength", 0.0)
        logger.info(f"🔁 [CYCLE] Alignment={cycle_alignment} | Strength={cycle_strength:.4f}")

        # 9️⃣ Bioclock Neuron
        if 'date' not in window_df.columns:
            window_df['date'] = window_df.index
        window_df['date'] = pd.to_datetime(window_df['date'], errors='coerce')
        bioclock = BioclockNeuron(
            df=window_df,
            phase_name=wave_phase,
            t_phase_start=self.spiral_wave_result.get("t_phase_start"),
            t_corrective_start=self.spiral_wave_result.get("t_corrective_start"),
            t_impulse_prior_duration=self.spiral_wave_result.get("t_impulse_prior_duration"),
            t_expected_phase_avg=self.spiral_wave_result.get("t_expected_phase_avg")
        )
        bioclock_output = bioclock.compute()
        peak_score = bioclock_output.get("peak_hours_score", 0.0)
        rhythm = bioclock_output.get("rhythm_alignment", "unknown")
        logger.info(f"⏰ [BIOCLOCK] Rhythm={rhythm} | Peak Score={peak_score:.4f}")

        # 🔟 Final Spiral Phase Detection (FESI-compliant)
        self.final_spiral_phase = finalize_spiral_phase({
            "entropy": entropy,
            "sei": sei,
            "sentiment": self.sentiment_score,
            "candle_pattern_score": float(self.candle_pattern_output.get("pattern_score", 0.0)),
            "trend_strength": trend_strength,
            "trend_state": trend_state,
            "cycle_strength": cycle_strength,
            "cycle_alignment": cycle_alignment,
            "bioclock_peak_score": peak_score,
            "rhythm_alignment": rhythm,
            "previous_phase": wave_phase,
            "wave_phase": wave_phase
        })
        logger.info(f"🔄 [PHASE] Finalized Spiral Phase: {self.final_spiral_phase}")

        # 1️⃣1️⃣ Epigenetic Gate Processing
        context = {
            "entropy_norm": entropy,
            "sei_norm": sei,
            "sentiment_norm": self.sentiment_score,
            "candle_pattern": self.candle_pattern_output,
            "spiral_phase": self.final_spiral_phase,
            "spiral_z_score": z_score,
            "spiral_fib_levels": self.spiral_wave_result.get("fib_levels", {}),
            "multi_tf_corr": self.multi_tf_corr_features,
            "multi_tf_indicators": self.multi_tf_indicator_features,
            "trend_neuron": self.trend_output,
            "cycle_neuron": cycle_output,
            "bioclock_neuron": bioclock_output,
            "df": window_df
        }
        gate = EpigeneticGate(phase_name=self.final_spiral_phase)
        self.gene_map, adjusted_vector = gate.evaluate(self.dna_vector, context, self.genome)
        logger.info(f"🧬 [EPI-GATE] Activated Genes: {list(self.gene_map.keys())}")

        # 1️⃣2️⃣ Neuron Cortex Activation
        cortex = NeuronProcessor(adjusted_vector, self.gene_map, context, self.organism, self.final_spiral_phase)
        self.neuron_outputs, _ = cortex.process()
        logger.info("🧠 [NEURONS] Firing complete — all neurons executed.")

        # 1️⃣3️⃣ Final State Vector
        builder = StateVectorBuilder(
            neuron_outputs=self.neuron_outputs,
            gene_map=self.gene_map,
            spiral_phase=self.final_spiral_phase,
            spiral_z=z_score,
            multi_tf_dfs=self.dfs
        )
        self.state_vector = builder.build_state_vector()
        logger.info(f"🧾 [STATE VECTOR] Completed | Shape={self.state_vector.shape}")
        logger.info(f"✅ [PIPELINE COMPLETE] Execution finished for {self.organism}.")

    def compute_multi_tf_correlation(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        closes = {}
        for tf, df in dfs.items():
            if "date" not in df.columns:
                df["date"] = df.index
            series = df.set_index("date")["close"]
            daily_series = series.resample("1D").last().ffill()
            closes[tf] = daily_series

        corr_matrix = np.corrcoef([closes[tf] for tf in self.timeframes])
        corr_features = {}
        for i in range(len(self.timeframes)):
            for j in range(i + 1, len(self.timeframes)):
                key = f"corr_{self.timeframes[i]}_{self.timeframes[j]}"
                try:
                    corr_features[key] = float(corr_matrix[i, j])
                except Exception:
                    corr_features[key] = 0.0
        return corr_features
