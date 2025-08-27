#!/usr/bin/env python3
# ==========================================================
# 🧠 WEPS NeuronProcessor v8.3 — Supreme Spiral Cortex (Final Production)
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Executes and stores ALL neuron outputs (reflex ready)
#   - Engineered for high-frequency decision processing in live markets
#   - Optimized for biological fidelity, precision, and spiral consistency
# ==========================================================

import logging
import pandas as pd

from weps.neurons import (
    bioclock_neuron, candle_pattern_neuron, correlation_neuron, cycle_neuron,
    elliott_wave_neuron, fibonacci_neuron, fractal_neuron, half_life_neuron,
    immune_response_neuron, impulse_neuron, indicator_neuron, liquidity_neuron,
    macro_sync_neuron, memory_trauma_neuron, metabolic_neuron, momentum_neuron,
    phase_transition_neuron, reflex_neuron, sentiment_neuron, spiral_density_neuron,
    support_resistance_neuron, trend_neuron, volatility_neuron, weakness_neuron
)

logger = logging.getLogger("WEPS.NeuronProcessor")

class NeuronProcessor:
    def __init__(self, dna_vector, active_genes, context, asset, phase):
        self.asset = asset
        self.phase = phase
        self.context = context
        self.df = context.get("df")
        self.dna_vector = dna_vector
        self.active_genes = active_genes or {}

        if not isinstance(self.df, pd.DataFrame):
            raise ValueError(f"[NeuronProcessor] Invalid DataFrame for {self.asset}")

        self.all_outputs = {}
        logger.info(f"[NeuronProcessor][{self.asset}][{self.phase}] Initializing spiral neuron cortex...")
        self._compute_all_outputs()

    def _compute_all_outputs(self):
        # === Wave + Cycle + Fibonacci
        try:
            self.all_outputs["elliott_wave"] = elliott_wave_neuron.ElliottWaveNeuron(self.df, self.phase).compute()
        except Exception as e:
            logger.warning(f"[{self.asset}] ElliottWaveNeuron failed: {e}")
            self.all_outputs["elliott_wave"] = None

        try:
            self.all_outputs["cycle"] = cycle_neuron.CycleNeuron(self.df, self.phase).compute(self.df)
        except Exception as e:
            logger.warning(f"[{self.asset}] CycleNeuron failed: {e}")
            self.all_outputs["cycle"] = None

        try:
            self.all_outputs["fibonacci"] = fibonacci_neuron.FibonacciNeuron(self.df, self.phase).compute(
                self.all_outputs["elliott_wave"],
                self.all_outputs["cycle"],
                None
            )
        except Exception as e:
            logger.warning(f"[{self.asset}] FibonacciNeuron failed: {e}")
            self.all_outputs["fibonacci"] = None

        # === Impulse / Trend / Volatility Scores
        impulse_score, trend_score, volatility_score = 0.0, 0.0, 0.0
        try:
            impulse_score = impulse_neuron.ImpulseNeuron(self.df, self.phase).compute().get("impulse_strength", 0.0)
        except: pass
        try:
            trend_score = trend_neuron.TrendNeuron(self.df, self.phase).compute().get("trend_strength", 0.0)
        except: pass
        try:
            volatility_score = volatility_neuron.VolatilityNeuron(self.df, self.phase).compute().get("volatility_score_norm", 0.0)
        except: pass

        # === Sentiment Signal
        try:
            sentiment_data = sentiment_neuron.SentimentNeuron(self.asset).compute(
                spiral_meta=self.context.get("spiral_meta", {}),
                wave_result=self.all_outputs.get("elliott_wave"),
                current_phase=self.phase
            )
            sentiment_signal = sentiment_data.get("sentiment_signal", "neutral")
            self.all_outputs["sentiment"] = sentiment_data
        except Exception as e:
            logger.warning(f"[{self.asset}] SentimentNeuron failed: {e}")
            sentiment_signal = "neutral"
            self.all_outputs["sentiment"] = {}

        # === Candle Pattern Detection
        try:
            candle_obj = candle_pattern_neuron.CandlePatternNeuron(self.phase)
            self.all_outputs["candle_pattern"] = candle_obj.compute(
                df=self.df,
                current_phase=self.phase
            )
        except Exception as e:
            logger.warning(f"[{self.asset}] CandlePatternNeuron failed: {e}")
            self.all_outputs["candle_pattern"] = {
                "pattern_name": "none",
                "pattern_class": "neutral",
                "pattern_score": 0.0,
                "spiral_phase": self.phase,
                "trap_signal": False
            }

        # === Main Neuron Map
        neuron_map = {
            "momentum": momentum_neuron.MomentumNeuron,
            "impulse": impulse_neuron.ImpulseNeuron,
            "trend": trend_neuron.TrendNeuron,
            "risk_defense": immune_response_neuron.ImmuneResponseNeuron,
            "weakness": weakness_neuron.WeaknessNeuron,
            "volatility": volatility_neuron.VolatilityNeuron,
            "bioclock": bioclock_neuron.BioclockNeuron,
            "indicator": indicator_neuron.IndicatorNeuron,
            "liquidity": liquidity_neuron.LiquidityNeuron,
            "correlation": correlation_neuron.CorrelationNeuron,
            "support_resistance": support_resistance_neuron.SupportResistanceNeuron,
            "half_life": half_life_neuron.HalfLifeNeuron,
            "fractal": fractal_neuron.FractalNeuron,
            "memory_trauma": memory_trauma_neuron.MemoryTraumaNeuron,
            "metabolic": metabolic_neuron.MetabolicNeuron,
            "macro_sync": macro_sync_neuron.MacroSyncNeuron,
            "spiral_density": spiral_density_neuron.SpiralDensityNeuron,
            "reflex": reflex_neuron.ReflexNeuron,
        }

        for name, cls in neuron_map.items():
            try:
                min_len = {
                    "momentum": 200, "liquidity": 250, "fractal": 300,
                    "memory_trauma": 500, "support_resistance": 200, "half_life": 200
                }.get(name, 0)

                if min_len and len(self.df) < min_len:
                    logger.warning(f"[{self.asset}] {name} neuron skipped: requires {min_len} rows")
                    self.all_outputs[name] = {}
                    continue

                if name == "indicator":
                    self.all_outputs[name] = cls(
                        self.df,
                        self.all_outputs.get("elliott_wave"),
                        self.all_outputs.get("candle_pattern"),
                        self.all_outputs.get("fibonacci"),
                        self.phase
                    ).compute()

                elif name == "correlation":
                    main_df = self.context.get("main_df")
                    sister_df = self.context.get("sister_df")
                    if isinstance(main_df, pd.DataFrame) and isinstance(sister_df, pd.DataFrame):
                        self.all_outputs[name] = cls(self.df, self.phase).compute(main_df, sister_df)
                    else:
                        self.all_outputs[name] = {}

                elif name == "risk_defense":
                    self.all_outputs[name] = cls(self.context, self.phase).compute()

                elif name == "weakness":
                    self.all_outputs[name] = cls(self.all_outputs.get("elliott_wave") or {}, self.phase).compute()

                else:
                    self.all_outputs[name] = cls(self.df, self.phase).compute()

            except Exception as e:
                logger.warning(f"[{self.asset}] {name} neuron failed: {e}")
                self.all_outputs[name] = {}

    def process(self):
        logger.info(f"[NeuronProcessor][{self.asset}] ✅ All neurons finalized. Outputs: {list(self.all_outputs.keys())}")
        return self.all_outputs, self.dna_vector
