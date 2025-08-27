#!/usr/bin/env python3
# ==========================================================
# 🔥 WEPS Spiral Policy Module — Institutional-Grade Decision Engine
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Combines spiral lifecycle signals, multi-timeframe correlation,
#     technical confirmations, sentiment adjustments, candle pattern confidence,
#     and confidence scoring
#   - Outputs a final trade decision with diagnostics
#   - Engineered for institutional rigor, robustness, and traceability
# ==========================================================
import logging
from typing import Dict, Any

logger = logging.getLogger("WEPS.SpiralPolicy")

# Define possible actions
ACTIONS = ["hold", "buy", "sell"]

def decide_trade(
    spiral_meta: Dict[str, Any],
    wave_result: Dict[str, Any],
    indicator_output: Dict[str, Any],
    sentiment_output: Dict[str, Any],
    multi_tf_data: Dict[str, Dict[str, Any]] = None,
    candle_pattern_output: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Decides trade action based on spiral intelligence signals, multi-TF confirmation,
    technical indicator alignment, sentiment adjustment, and candle pattern confidence.

    Args:
        spiral_meta: Latest spiral meta data (phase, z_score, entropy_slope, etc.)
        wave_result: Wave engine output (phase, direction, validated, z_score, fib_levels)
        indicator_output: Indicator neuron outputs (long_ema_slope_norm, rsi, adx, etc.)
        sentiment_output: Sentiment neuron output (sentiment_signal, sentiment_adjustment, ...)
        multi_tf_data: Optional dict keyed by timeframe containing similar dicts of signals
                       to perform multi-timeframe confluence checks.
        candle_pattern_output: Optional dict containing candle pattern info:
            {
                "pattern": str,
                "transition": str
            }

    Returns:
        Dict with keys:
            - action (str): 'buy', 'sell', or 'hold'
            - confidence (float): 0.0 to 1.0 confidence level
            - reason (str): diagnostic message explaining decision
            - diagnostics (dict): detailed scoring breakdown and input echo
    """

    # Defensive default output
    output = {
        "action": "hold",
        "confidence": 0.0,
        "reason": "Default hold - insufficient or conflicting signals",
        "diagnostics": {}
    }

    try:
        phase = spiral_meta.get("phase", "unknown").lower()
        z_score = float(spiral_meta.get("z_score", 0.0))
        entropy = float(spiral_meta.get("entropy_slope", 0.0))
        sei = float(spiral_meta.get("sei_slope", 0.0))

        wave_phase = wave_result.get("phase", "unknown").lower()
        wave_direction = wave_result.get("direction", "neutral").lower()
        wave_validated = wave_result.get("validated", False)

        indicator_confirm = float(indicator_output.get("long_ema_slope_norm", 0.0))
        rsi = indicator_output.get("indicator_state", {}).get("rsi", 50)
        adx = indicator_output.get("indicator_state", {}).get("adx", 20)

        sentiment_adj = float(sentiment_output.get("sentiment_adjustment", 0.0))
        sentiment_signal = sentiment_output.get("sentiment_signal", "neutral").lower()

        # Candle pattern integration
        pattern = candle_pattern_output.get("pattern", "unknown") if candle_pattern_output else "unknown"
        transition = candle_pattern_output.get("transition", "unknown") if candle_pattern_output else "unknown"

        candle_confidence = 0.0
        if pattern in ["bullish_engulfing", "hammer", "morning_star"]:
            candle_confidence += 0.1
        elif pattern in ["bearish_engulfing", "shooting_star", "evening_star"]:
            candle_confidence -= 0.1

        # Multi-Timeframe Confirmation: count agreeing phases & directions if multi_tf_data provided
        multi_tf_agree_count = 0
        multi_tf_total = 0
        if multi_tf_data:
            for tf, data in multi_tf_data.items():
                multi_tf_total += 1
                tf_phase = data.get("phase", "unknown").lower()
                tf_direction = data.get("direction", "neutral").lower()
                if tf_phase == phase and tf_direction == wave_direction:
                    multi_tf_agree_count += 1

        multi_tf_agree_ratio = multi_tf_agree_count / multi_tf_total if multi_tf_total > 0 else 1.0

        # Basic Spiral Phase-to-Action mapping
        action_bias = 0
        if phase == "growth" and wave_direction == "bullish":
            action_bias += 1
        elif phase == "decay" and wave_direction == "bearish":
            action_bias -= 1
        elif phase == "rebirth":
            action_bias += 0  # Neutral

        # Indicator confirmations: EMA slope + RSI + ADX
        indicator_score = 0
        indicator_score += indicator_confirm  # normalized EMA slope
        indicator_score += (rsi - 50) / 50.0  # RSI deviation from midline normalized [-1,1]
        indicator_score += (adx / 50.0)       # ADX scaled (max 50 considered strong trend)
        indicator_score = max(min(indicator_score / 3.0, 1.0), -1.0)  # normalize final to [-1,1]

        # Sentiment adjustment weight
        sentiment_weight = 0.15
        sentiment_score = 0
        if sentiment_signal == "positive":
            sentiment_score = sentiment_weight
        elif sentiment_signal == "negative":
            sentiment_score = -sentiment_weight

        # Combine scores for final confidence:
        combined_score = (action_bias * 0.5) + (indicator_score * 0.3) + (sentiment_score) + (multi_tf_agree_ratio * 0.2)
        combined_score += candle_confidence
        combined_score = max(min(combined_score, 1.0), -1.0)

        # Final confidence as absolute combined_score (scaled 0-1)
        confidence = abs(combined_score)

        # Determine action by sign
        if combined_score > 0.3 and confidence > 0.5:
            action = "buy"
        elif combined_score < -0.3 and confidence > 0.5:
            action = "sell"
        else:
            action = "hold"

        reason = (
            f"SpiralPhase={phase}, WaveDir={wave_direction}, Validated={wave_validated}, "
            f"ZScore={z_score:.3f}, Entropy={entropy:.4f}, IndicatorScore={indicator_score:.3f}, "
            f"SentimentAdj={sentiment_score:.3f}, MultiTFConfirm={multi_tf_agree_ratio:.2f}, "
            f"CandleConf={candle_confidence:.3f}, CombinedScore={combined_score:.3f}, Confidence={confidence:.3f}, "
            f"CandlePattern={pattern}, Transition={transition}"
        )

        diagnostics = {
            "spiral_meta": spiral_meta,
            "wave_result": wave_result,
            "indicator_output": indicator_output,
            "sentiment_output": sentiment_output,
            "multi_tf_agree_ratio": multi_tf_agree_ratio,
            "combined_score": combined_score,
            "confidence": confidence,
            "action_bias": action_bias,
            "indicator_score": indicator_score,
            "sentiment_score": sentiment_score,
            "candle_confidence": candle_confidence,
            "candle_pattern": pattern,
            "candle_transition": transition,
        }

        output.update({
            "action": action,
            "confidence": confidence,
            "reason": reason,
            "diagnostics": diagnostics
        })

        logger.warning(f"[SpiralPolicy] Decision: {action.upper()} Confidence={confidence:.3f} | Reason: {reason}")

        return output

    except Exception as ex:
        logger.error(f"[SpiralPolicy] Exception in decide_trade(): {ex}", exc_info=True)
        return {
            "action": "hold",
            "confidence": 0.0,
            "reason": f"Exception in SpiralPolicy: {ex}",
            "diagnostics": {}
        }
