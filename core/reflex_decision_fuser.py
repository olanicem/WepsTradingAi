#!/usr/bin/env python3
# ==========================================================
# 🧠 WEPS Reflex Cortex Decision Fuser (v3)
# ✅ Spiral Phase Directional Awareness + Timing
# ✅ 5/6 Indicator Enforcement + ReflexPolicy Integration
# ✅ Final Trade Decision with Robust Fallbacks
# ==========================================================
import numpy as np
import torch
from weps.core.reflex_policy_engine import ReflexPolicyEngine

def fuse_trade_decision(
    state_seq: torch.Tensor,
    prior_impulsion_distance: float,
    sentiment_score: float,
    corrective_wave_duration: float,
    atr: float,
    price_data: dict,
    indicators: dict,
    z_score_main: float,
    z_score_sub: float,
    z_score_micro: float,
    corrective_wave_timing_factor: float,
    live_hit_rate: float = 0.95,
    state_quality_score: float = 1.0,
    atr_volatility_factor: float = 1.0,
    common_retrace_levels: dict = None,
    phase_shift_score_norm: float = 0.0,
    depth_in_phase: float = 0.0,
    phase_name: str = "unknown",
    momentum: float = 0.0,
) -> dict:
    """
    Advanced WEPS3-EPTS Decision Fusion:
    - Spiral phase determines direction.
    - Corrective timing & 5/6 indicator rule enforced.
    - ReflexPolicyEngine enriches levels.
    """

    # 🌐 Spiral Phase Direction Logic
    if phase_name in ["rebirth", "growth"]:
        if momentum > 0 and z_score_main >= 0.85:
            action = "buy"
        elif momentum < 0 and z_score_main >= 0.85:
            action = "sell"
        else:
            action = "hold"
    elif phase_name == "decay":
        if momentum < 0 and z_score_main >= 0.85:
            action = "sell"
        elif momentum > 0 and z_score_main >= 0.85:
            action = "buy"
        else:
            action = "hold"
    else:
        action = "hold"

    # ⚡ Enforce Corrective Timing Factor & 5/6 Indicators
    indicator_confirmations = sum([
        indicators.get("macd", False),
        indicators.get("rsi", False),
        indicators.get("stoch", False),
        indicators.get("adx", False),
        indicators.get("alligator", False),
        indicators.get("volume", False),
    ])
    if corrective_wave_timing_factor < 1 or indicator_confirmations < 5:
        action = "hold"

    # 🛡️ Fallback on corrupted z-scores
    if np.isnan(z_score_main) or np.isinf(z_score_main) or abs(z_score_main) > 10:
        action = "hold"

    # 🔄 Compute enriched levels using ReflexPolicyEngine
    policy = ReflexPolicyEngine(
        prior_impulsion_distance=prior_impulsion_distance,
        sentiment_score=sentiment_score,
        corrective_wave_duration=corrective_wave_duration,
        atr=atr,
        price_data=price_data,
        indicators=indicators,
        state_vector={},
        q_values=torch.tensor([[0.0, 0.0, 0.0, 0.0]]),  # dummy
        z_score_main=z_score_main,
        z_score_sub=z_score_sub,
        z_score_micro=z_score_micro,
        corrective_wave_timing_factor=corrective_wave_timing_factor,
        live_hit_rate=live_hit_rate,
        state_quality_score=state_quality_score,
        atr_volatility_factor=atr_volatility_factor,
        common_retrace_levels=common_retrace_levels,
        phase_shift_score_norm=phase_shift_score_norm,
        depth_in_phase=depth_in_phase,
    )
    levels = policy.compute_levels()

    current_close = price_data.get("current_close", 0.0)
    entry_point = (
        levels["retraces"].get("fib_382") or
        levels["retraces"].get("fib_50") or
        current_close
    ) if levels else current_close

    stop_loss = (
        levels["retraces"].get("fib_618") if action == "buy" else
        levels["retraces"].get("fib_382")
    ) if levels else (
        entry_point - (2 * atr) if action == "buy" else entry_point + (2 * atr)
    )

    take_profit = (
        levels["extensions"].get("elliot_target_127")
        if levels else (
            entry_point + prior_impulsion_distance * 0.464 if action == "buy"
            else entry_point - prior_impulsion_distance * 0.464
        )
    )

    # 🔎 Confidence & Probability Calculations
    trade_quality_factor = (
        live_hit_rate * 0.4 +
        state_quality_score * 0.25 +
        min(z_score_main, 1.0) * 0.15 +
        min(z_score_sub, 1.0) * 0.1 +
        min(z_score_micro, 1.0) * 0.07 +
        max(1.0 - abs(sentiment_score), 0.5) * 0.03
    )
    confidence_score = round(100 * trade_quality_factor, 2)
    hit_probability_for_Y = round(100 * trade_quality_factor * atr_volatility_factor, 2)

    # 📝 Return Final Decision
    return {
        "action": action,
        "entry_point": round(entry_point, 6),
        "stop_loss": round(stop_loss, 6),
        "take_profit": round(take_profit, 6),
        "exit_timing": round(corrective_wave_duration * 0.63 * (1.0 - phase_shift_score_norm * 0.1), 4),
        "confidence_score": confidence_score,
        "hit_probability_for_Y": f">{hit_probability_for_Y}%",
        "z_score_main": z_score_main,
        "z_score_sub": z_score_sub,
        "z_score_micro": z_score_micro,
        "phase_name": phase_name,
        "momentum": momentum,
        "indicator_confirmations": indicator_confirmations,
        "corrective_wave_timing_factor": corrective_wave_timing_factor,
        "reason": "Trade decision fused with spiral phase, momentum, timing factor, and 5/6 indicators."
    }
