#!/usr/bin/env python3
# ==========================================================
# 🌀 WEPS Supreme Spiral Phase Detector — FESI++ Final Edition
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Full Biologically Reinforced + Wave-Fused Spiral Phase Resolver
#   - Uses entropy, SEI, impulse, volatility, gene activation, z-score, fib alignment
#   - Phase continuity, hysteresis, conflict dampening, and reflex-aware transitions
# ==========================================================

import logging
from typing import Dict, Any

logger = logging.getLogger("WEPS.SpiralPhaseDetector")

PHASES = ["rebirth", "growth", "decay", "death", "neutral"]

def detect_spiral_phase(context: Dict[str, Any]) -> str:
    logger.info("[SpiralPhaseDetector] 🧬 Starting full biological phase analysis (FESI++).")

    # === Core biological signals ===
    entropy = float(context.get("entropy", 50.0))              # 0–100
    sei = float(context.get("sei", 50.0))                      # 0–100
    impulse = float(context.get("impulse", 0.0))               # 0–1
    momentum = float(context.get("momentum", 0.0))             # 0–1
    volatility = float(context.get("volatility", 0.5))         # 0–1
    destruction = float(context.get("destruction", 0.0))       # 0–1

    # === Gene activations ===
    correction_gene = float(context.get("correction_gene", 0.5))
    weakness_gene = float(context.get("weakness_gene", 0.5))
    risk_defense_gene = float(context.get("risk_defense_gene", 0.5))
    bioclock = float(context.get("bioclock", 0.5))

    # === Auxiliary signal scores ===
    z_score = float(context.get("spiral_z_score", 0.0))                  # wave score
    fib_alignment = float(context.get("fib_alignment_score", 0.5))      # 0–1
    candle_score = float(context.get("candle_pattern_score", 0.0))      # 0–1
    phase_duration_score = float(context.get("phase_duration_score", 0.5))  # 0–1
    trend_state = context.get("trend_state", "unknown")
    cycle_alignment = context.get("cycle_alignment", "unknown")

    previous_phase = context.get("previous_phase", "neutral")

    # === Derived scoring logic ===
    chaotic_flux = volatility * (entropy / 100.0)

    rebirth_score = (
        max(0.0, 1.0 - entropy / 55.0) *
        impulse *
        momentum *
        (1.0 - destruction) *
        (1.0 - correction_gene) *
        (0.6 + 0.4 * fib_alignment)
    )

    growth_score = (
        max(0.0, 1.0 - abs(entropy - 50.0) / 50.0) *
        momentum *
        impulse *
        (1.0 - risk_defense_gene) *
        (1.0 if trend_state in ["bullish", "uptrend"] else 0.7) *
        (0.6 + 0.4 * candle_score)
    )

    decay_score = (
        (entropy / 100.0) *
        (1.0 - momentum) *
        (1.0 - impulse) *
        ((weakness_gene + correction_gene) / 2.0) *
        (0.7 + 0.3 * (1.0 if cycle_alignment in ["bearish", "misaligned"] else 0.5))
    )

    death_score = (
        max(0.0, (entropy - 70.0) / 30.0) *
        destruction *
        (1.0 - impulse) *
        (1.0 - momentum) *
        risk_defense_gene *
        (0.5 + 0.5 * chaotic_flux)
    )

    scores = {
        "rebirth": rebirth_score,
        "growth": growth_score,
        "decay": decay_score,
        "death": death_score,
        "neutral": 0.1
    }

    # === Normalize + Continuity Boost ===
    total = sum(scores.values())
    if total == 0.0:
        logger.warning("[SpiralPhaseDetector] ⚠️ All biological scores are zero. Returning 'neutral'.")
        return "neutral"

    norm = {k: v / total for k, v in scores.items()}

    if previous_phase in norm:
        norm[previous_phase] *= 1.2  # hysteresis: stabilize last phase

    # === Final Decision ===
    final_phase = max(norm.items(), key=lambda x: x[1])[0]

    logger.info(f"[SpiralPhaseDetector] 🧠 Score distribution: {norm}")
    logger.info(f"[SpiralPhaseDetector] ✅ Selected biological phase: {final_phase} (prior: {previous_phase})")

    return final_phase


def finalize_spiral_phase(context: Dict[str, Any]) -> str:
    """
    Final phase logic:
    - If wave phase is clean + matches bio estimate, it's accepted.
    - If wave phase contradicts biological markers, perform override.
    - Fully reflex-aware, entropy-weighted final judgment.
    """
    wave_phase = context.get("wave_phase", "unknown")
    previous = context.get("previous_phase", "neutral")
    context["previous_phase"] = previous

    # === Biological Score
    bio_phase = detect_spiral_phase(context)

    # === Sanity Check: Validate Wave Phase
    if wave_phase in PHASES and wave_phase != "unknown":
        entropy = context.get("entropy", 50.0)
        impulse = context.get("impulse", 0.5)
        destruction = context.get("destruction", 0.5)

        # 🌪 Contradiction Check
        if wave_phase == "rebirth" and entropy > 80:
            logger.warning("[SpiralPhaseDetector] ⛔ Wave phase 'rebirth' rejected: entropy too high.")
        elif wave_phase == "death" and impulse > 0.6:
            logger.warning("[SpiralPhaseDetector] ⛔ Wave phase 'death' rejected: impulse too strong.")
        else:
            logger.info(f"[SpiralPhaseDetector] 🌀 Wave phase confirmed: {wave_phase}")
            return wave_phase

    # === Override with Biological Estimate
    logger.info(f"[SpiralPhaseDetector] 🧬 Overriding wave phase. Final: {bio_phase}")
    return bio_phase
