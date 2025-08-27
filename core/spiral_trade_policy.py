#!/usr/bin/env python3
# ==============================================================
# ⚖️ Spiral Reflex Policy — Unified Adaptive Decision Engine
# ✅ Integrates Spiral Phase, Wave Z, Entropy, Indicators, and Sentiment
# ✅ Sets action, dynamic thresholds, adaptive confidence, and logs reasoning
# Author: Ola Bode (WEPS Creator)
# ==============================================================
import logging

logger = logging.getLogger("WEPS.SpiralPolicy")


def decide_trade(spiral_meta, wave_result, indicator_output, sentiment_output):
    """
    Computes final trade decision using FESI spiral intelligence principles:
      - Spiral phase, wave z-score, entropy regime
      - IndicatorNeuron confluence score
      - SentimentNeuron dynamic bias adjustment
    Returns final trade command dict with reasoning.
    """
    phase = spiral_meta.get("phase", "unknown")
    spiral_z = spiral_meta.get("z_score", 0.0)
    entropy = spiral_meta.get("entropy_slope", 1.0)

    # IndicatorNeuron-based confirmations → confidence component
    confirmations = indicator_output.get("long_ema_slope_norm", 0.0) if indicator_output else 0.0

    # SentimentNeuron adjustment → confidence component
    sentiment_adj = sentiment_output.get("sentiment_adjustment", 0.0) if sentiment_output else 0.0

    # Compute base confidence using spiral z and entropy regime
    base_confidence = max(0.0, min(1.0, (spiral_z * (1 - entropy)) + confirmations))
    confidence = round(base_confidence + sentiment_adj, 4)

    # Define action logic adaptively with spiral phase
    action = "hold"
    if phase in ["rebirth", "growth"] and spiral_z > 0.3 and entropy < 0.005 and confirmations > 0.6:
        action = "buy"
    elif phase == "decay" and spiral_z < 0.3 and entropy > 0.005 and confirmations < 0.4:
        action = "sell"

    # Compose diagnostic reason string
    reason = (
        f"SpiralPolicy → phase={phase}, z={spiral_z:.3f}, entropy={entropy:.6f}, "
        f"confirmations={confirmations:.3f}, sentiment_adj={sentiment_adj:.3f}"
    )

    # Log the complete decision details at institutional grade
    logger.warning(
        f"[SpiralPolicy] Phase={phase} Z={spiral_z:.4f} Entropy={entropy:.6f} "
        f"Confirmations={confirmations:.4f} SentimentAdj={sentiment_adj:.4f} → "
        f"Decision={action.upper()} Confidence={confidence:.4f}"
    )

    return {
        "action": action,
        "confidence": confidence,
        "reason": reason
    }
