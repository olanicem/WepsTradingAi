#!/usr/bin/env python3
# ==========================================================
# 🧬 WEPS Spiral-Intelligent Genome — Supreme Production Version
# ✅ Fully Spiral & Phase-Aware | Z-Scores | Elliott+Fibonacci
# ✅ Integrates All Neurons into True Living Market DNA
# ✅ Uses Precomputed Candle Patterns for Efficiency & Accuracy
# ✅ Generates Adaptive Genome for Reflex Cortex & AI Agents
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import logging
from weps.neurons.indicator_neuron import IndicatorNeuron
from weps.neurons.metabolism_neuron import MetabolismNeuron
from weps.neurons.momentum_neuron import MomentumNeuron
from weps.neurons.phase_transition_neuron import PhaseTransitionNeuron
from weps.neurons.sentiment_neuron import SentimentNeuron
from weps.neurons.correlation_neuron import CorrelationNeuron
from weps.neurons.half_life_neuron import HalfLifeNeuron
from weps.neurons.immune_response_neuron import ImmuneResponseNeuron
from weps.neurons.macro_sync_neuron import MacroSyncNeuron
from weps.neurons.memory_trauma_neuron import MemoryTraumaNeuron

logger = logging.getLogger("WEPS.Genome")

class WEPSGenome:
    """
    WEPS Genome
    - Spiral-aware living genome blueprint for market organisms.
    - Encodes Z-scores, phase-aligned neuron signals, and wave validations.
    - Outputs adaptive genetic profile for Reflex Cortex and AI trading engines.
    """
    def __init__(self, df, wave_results, elliott_result, fib_result, df_sister=None, candle_result=None):
        """
        Args:
            df (pd.DataFrame): Main asset data.
            wave_results (dict): Spiral wave analysis output.
            elliott_result (dict): Elliott wave neuron results.
            fib_result (dict): Fibonacci neuron results.
            df_sister (pd.DataFrame|None): Sister asset data for correlation.
            candle_result (dict|None): Precomputed CandlePatternNeuron result.
        """
        self.df = df
        self.wave_results = wave_results
        self.elliott_result = elliott_result
        self.fib_result = fib_result
        self.df_sister = df_sister
        self.candle_result = candle_result

        # Updated: read from flat wave_results output
        self.z_score_main = wave_results["z_score"]
        self.z_score_sub = wave_results["z_score"]
        self.z_score_micro = wave_results["z_score"]
        self.phase_name = wave_results["spiral_phase"]

        logger.info("✅ WEPSGenome initialized: Z-main=%.4f, Z-sub=%.4f, Z-micro=%.4f, phase=%s",
                    self.z_score_main, self.z_score_sub, self.z_score_micro, self.phase_name)

    def to_dict(self) -> dict:
        price_data = {
            "open": float(self.df["open"].iloc[-1]),
            "high": float(self.df["high"].iloc[-1]),
            "low": float(self.df["low"].iloc[-1]),
            "close": float(self.df["close"].iloc[-1]),
            "volume": int(self.df["volume"].iloc[-1]),
        }

        # Compute spiral-aware neurons
        indicators = IndicatorNeuron(self.df, self.phase_name).compute(self.df)
        metabolism = MetabolismNeuron(self.df, self.phase_name).compute(self.df)
        momentum = MomentumNeuron(self.df, self.phase_name).compute(self.df)
        phase_transition = PhaseTransitionNeuron(self.df, self.phase_name).compute(self.df)
        sentiment = SentimentNeuron(self.df, self.phase_name).compute(self.df)
        correlation = (CorrelationNeuron(self.df, self.df_sister, self.phase_name).compute()
                       if self.df_sister is not None else None)
        half_life = HalfLifeNeuron(self.df, self.phase_name).compute(self.df)
        immune_response = ImmuneResponseNeuron(self.df, self.phase_name).compute(self.df)
        macro_sync = MacroSyncNeuron(self.df, self.phase_name).compute(self.df)
        memory_trauma = MemoryTraumaNeuron(self.df, self.phase_name).compute(self.df)

        # Use precomputed candle result for accuracy and speed
        candle_pattern = self.candle_result

        return {
            "price": price_data,
            "spiral_phase": self.phase_name,
            "z_scores": {
                "main": self.z_score_main,
                "sub": self.z_score_sub,
                "micro": self.z_score_micro
            },
            "indicators": indicators,
            "contextual_state": {
                "metabolism": metabolism,
                "momentum": momentum,
                "phase_transition": phase_transition,
                "sentiment": sentiment,
                "correlation": correlation,
                "half_life": half_life,
                "immune_response": immune_response,
                "macro_sync": macro_sync,
                "memory_trauma": memory_trauma,
                "candle_pattern": candle_pattern,
                "elliott_wave": self.elliott_result,
                "fibonacci": self.fib_result,
            }
        }
