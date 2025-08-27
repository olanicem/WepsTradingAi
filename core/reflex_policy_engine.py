#!/usr/bin/env python3
# ==============================================================
# 🧬 WEPS Reflex Cortex v5.1 — Immortal Spiral Trade Decision Engine
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Combines bio-spiral phases, entropy, SEI, z-scores, candle neurons
#   - Reflex-level override protections with evolutionary learning structure
#   - Interface-safe neuron schema to prevent malformed data usage
# ==============================================================

import numpy as np
import logging

from weps.core.spiral_trade_executor import SpiralTradeExecutor
from weps.utils.log_utils import log_spiral_decision

logger = logging.getLogger("WEPS.ReflexPolicyEngine")


class ReflexPolicyEngine:
    def __init__(self, spiral_vector, wave_result, df, entropy_slope,
                 neuron_processor, spiral_phase, x, sentiment_output, organism):
        self.spiral_vector = spiral_vector
        self.wave_result = wave_result
        self.df = df
        self.entropy = entropy_slope
        self.neuron_processor = neuron_processor
        self.spiral_phase = spiral_phase
        self.x = x
        self.sentiment_output = sentiment_output or {}
        self.organism = organism

    def decide(self):
        try:
            entry_price = self.df["close"].iloc[-1]
            atr = max((self.df["high"].rolling(14).max() - self.df["low"].rolling(14).min()).iloc[-1], 0.01)

            # 🌀 Spiral Biometrics
            sei = np.clip(np.std(self.spiral_vector), 0.0, 1.0)
            entropy_score = 1.0 - np.clip(self.entropy, 0.0, 1.0)
            z_main = self.wave_result.get("z_score", 0.0)
            z_micro = self.wave_result.get("z_score_micro", 0.0)
            spiral_depth = self.wave_result.get("spiral_depth", 0.5)
            timing_score = np.clip(self.x, 0.0, 2.0)
            sentiment_adj = self.sentiment_output.get("sentiment_adjustment", 0.0)

            # 🔍 Directional Alignment
            trend_bias = "buy" if self.spiral_phase in ["rebirth", "growth"] else "sell"

            executor = SpiralTradeExecutor(self.organism)
            spiral_meta = {
                "spiral_phase": self.spiral_phase,
                "sei": sei,
                "entropy": 1.0 - entropy_score,
                "z_score": z_main,
                "bioclock_peak": timing_score
            }
            trade_decision = executor.evaluate_trade_readiness(spiral_meta, trend_direction=trend_bias)
            log_spiral_decision(self.organism, trade_decision)

            # 🧠 Neuron Confidence
            all_neurons = self.neuron_processor.all_neurons

            # ✅ WEPS-Grade Neuron Schema Guard
            required_genes = ["candle", "impulse", "trend", "volatility", "cycle"]
            for gene in required_genes:
                if gene not in all_neurons:
                    logger.warning(f"[ReflexCortex] ⚠️ Missing expected neuron: {gene}")
                    continue
                neuron = all_neurons[gene]
                if not hasattr(neuron, "__class__"):
                    logger.warning(f"[ReflexCortex] ⚠️ Invalid neuron structure for gene: {gene}")
                    continue
                if hasattr(neuron, "validate") and callable(neuron.validate):
                    try:
                        neuron.validate()
                    except Exception as ve:
                        logger.warning(f"[ReflexCortex] ⚠️ Validation failed for neuron {gene}: {ve}")

            support_genes = [g for g, n in all_neurons.items() if hasattr(n, "confirms_phase") and n.confirms_phase()]
            reject_genes = [g for g, n in all_neurons.items() if hasattr(n, "rejects_phase") and n.rejects_phase()]
            neuron_strength = len(support_genes) / len(all_neurons) if all_neurons else 0.0

            # 🕯 Candle Override (Safe Access)
            candle = all_neurons.get("candle", None)
            if candle and isinstance(candle, object):
                candle_aligned = getattr(candle, "pattern_aligned", False)
                candle_strength = getattr(candle, "pattern_strength", 0.0)
                candle_name = getattr(candle, "pattern_name", None)
            else:
                candle_aligned = False
                candle_strength = 0.0
                candle_name = None

            # 🧠 Confidence Score
            raw_conf = (
                0.25 * z_main +
                0.25 * neuron_strength +
                0.20 * sei +
                0.15 * entropy_score +
                0.10 * timing_score +
                0.05 * sentiment_adj
            )

            # 📊 Candle Adjustments
            if candle_name:
                if self.spiral_phase in ['rebirth', 'growth']:
                    if candle_aligned and z_micro > 0.2:
                        raw_conf += 0.07 * candle_strength
                    elif not candle_aligned and z_micro < 0.1:
                        raw_conf -= 0.07 * candle_strength
                elif self.spiral_phase in ['decay', 'death']:
                    if not candle_aligned and z_micro < 0.15:
                        raw_conf = min(raw_conf, 0.6)
                    elif candle_aligned and z_micro < 0.1:
                        raw_conf = min(raw_conf, 0.5)

            final_conf = np.clip(raw_conf, 0, 1)
            action = trade_decision["decision"]
            if final_conf < 0.72 or len(reject_genes) > len(support_genes):
                action = "hold"

            # 🧬 Dynamic SL/TP
            sl = round(entry_price - (atr * 1.5 * (1 if action == "buy" else -1)), 5)
            tp = round(entry_price + (atr * 3.5 * (1 if action == "buy" else -1)), 5)

            # 📈 Fibonacci
            fibs = self.wave_result.get("fib_levels", {})
            fib_382 = float(fibs.get("0.382", 0.0))
            fib_618 = float(fibs.get("0.618", 0.0))
            fib_1618 = float(fibs.get("1.618", 0.0))

            # 📦 Final Output
            return {
                "action": action,
                "phase": self.spiral_phase,
                "confidence": round(final_conf, 4),
                "entry": round(entry_price, 5),
                "sl": sl,
                "tp": tp,
                "spiral_z": round(z_main, 4),
                "entropy": round(self.entropy, 6),
                "sei": round(sei, 4),
                "x_timing": round(timing_score, 4),
                "sentiment_signal": self.sentiment_output.get("sentiment_signal", "neutral"),
                "support_genes": support_genes,
                "reject_genes": reject_genes,
                "fib_levels": {
                    "0.382": fib_382,
                    "0.618": fib_618,
                    "1.618": fib_1618
                },
                "reflex_breakdown": {
                    "z_main": round(z_main, 4),
                    "neuron_strength": round(neuron_strength, 4),
                    "sei": round(sei, 4),
                    "entropy_score": round(entropy_score, 4),
                    "sentiment_adj": round(sentiment_adj, 4),
                    "timing_score": round(timing_score, 4),
                    "z_micro": round(z_micro, 4),
                    "candle_override": {
                        "applied": bool(candle_name),
                        "aligned": candle_aligned,
                        "name": candle_name,
                        "strength": round(candle_strength, 2),
                        "adjustment": round(
                            0.07 * candle_strength if candle_aligned and self.spiral_phase in ["rebirth", "growth"]
                            else -0.07 * candle_strength if not candle_aligned and self.spiral_phase in ["rebirth", "growth"]
                            else 0.0, 4
                        )
                    }
                }
            }

        except Exception as e:
            logger.error(f"[ReflexCortex] ❌ Critical error: {str(e)}")
            return {
                "action": "hold",
                "phase": self.spiral_phase,
                "confidence": 0.0,
                "error": str(e),
                "fallback": True
            }


__all__ = ["ReflexPolicyEngine"]
