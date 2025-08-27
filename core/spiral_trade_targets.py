#!/usr/bin/env python3
# ==============================================================
# ⚡ Spiral Trade Target Computation — Standalone Utility
# ✅ Computes dynamic entry, exit, SL, TP with SEI & Entropy-based Spiral Weighting
# ✅ Optional integration of Fibonacci levels for enhanced precision
# ✅ Fully production-hardened, with clear logs & error resilience
# Author: Ola Bode (WEPS Creator)
# ==============================================================

import numpy as np
import logging

logger = logging.getLogger("WEPS.SpiralTradeTargets")

def compute_spiral_trade_targets(
    entry_base: float,
    atr: float,
    sei_slope: float,
    entropy_slope: float,
    spiral_depth: float,
    direction: int = 1,
    fib_0_382: float = None,
    fib_0_618: float = None,
    fib_1_618: float = None
) -> dict:
    """
    Computes spiral-aware entry, exit, stop-loss, and take-profit targets.

    Args:
        entry_base (float): Current close price as entry reference.
        atr (float): Average True Range for volatility scaling.
        sei_slope (float): Slope of SEI (Spiral Energy Index), normalized.
        entropy_slope (float): Slope of entropy, normalized.
        spiral_depth (float): Current spiral depth phase (0-1).
        direction (int): 1 for long, -1 for short.
        fib_0_382 (float, optional): Optional Fibonacci 38.2% retracement.
        fib_0_618 (float, optional): Optional Fibonacci 61.8% retracement.
        fib_1_618 (float, optional): Optional Fibonacci 161.8% extension.

    Returns:
        dict: Dictionary with computed 'entry', 'exit', 'stop_loss', 'take_profit'.
    """
    try:
        spiral_weight = 1 + spiral_depth + sei_slope - entropy_slope
        spiral_weight = np.clip(spiral_weight, 0.5, 2.0)

        take_profit = entry_base + direction * (atr * 2 * spiral_weight)
        stop_loss = entry_base - direction * (atr * 1.5 * spiral_weight)
        exit = entry_base + direction * (atr * 0.63 * spiral_weight)

        # If Fibonacci levels are valid, blend into spiral targets
        if fib_0_382 is not None and fib_0_618 is not None and fib_1_618 is not None:
            take_profit = (take_profit + fib_1_618) / 2
            exit = (exit + fib_0_618) / 2
            stop_loss = (stop_loss + fib_0_382) / 2

        logger.info("[SpiralTargets] Computed targets → entry=%.5f exit=%.5f TP=%.5f SL=%.5f spiral_weight=%.3f",
                    entry_base, exit, take_profit, stop_loss, spiral_weight)

        return {
            "entry": round(entry_base, 5),
            "exit": round(exit, 5),
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
        }

    except Exception as e:
        logger.error("[SpiralTargets] Error computing spiral trade targets: %s", str(e), exc_info=True)
        raise e
