#!/usr/bin/env python3
# ================================================================
# 🧬 WEPS Spiral Trade Executor — Reflex-Intelligent Core (FESI Compliant)
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Biologically intelligent trade execution logic based on:
#     Spiral Phase, SEI, Entropy, Z-score, and BioClock
#   - Supports both BUY (rebirth/growth) and SELL (decay/death)
# ================================================================

from typing import Dict, Optional

class SpiralTradeExecutor:
    def __init__(self, organism: str):
        self.organism = organism

    def evaluate_trade_readiness(self, spiral_data: Dict[str, float], trend_direction: Optional[str]) -> Dict[str, any]:
        """
        Decide whether to buy/sell/hold based on WEPS spiral intelligence logic.

        Args:
            spiral_data: {
                'spiral_phase': str,
                'sei': float,
                'entropy': float,
                'z_score': float,
                'bioclock_peak': float
            }
            trend_direction: 'buy' | 'sell' | 'hold' | None

        Returns:
            Decision dictionary with trade permission and signal.
        """
        phase = spiral_data.get("spiral_phase", "").lower()
        sei = spiral_data.get("sei", 0.0)
        entropy = spiral_data.get("entropy", 1.0)
        z_score = spiral_data.get("z_score", 0.0)
        bioclock_peak = spiral_data.get("bioclock_peak", 0.0)

        # ✅ Long Trade Conditions (Growth)
        long_allowed = (
            phase in ['rebirth', 'growth']
            and sei > 0.3
            and entropy < 0.35
            and -2.5 <= z_score <= 0.5
            and bioclock_peak >= 0.7
        )

        # ✅ Short Trade Conditions (Collapse)
        short_allowed = (
            phase in ['decay', 'death']
            and sei < 0.4
            and entropy > 0.35
            and z_score > 1.5
            and bioclock_peak >= 0.7
        )

        # 🔁 Final Decision
        decision = "hold"
        if long_allowed and trend_direction == "buy":
            decision = "buy"
        elif short_allowed and trend_direction == "sell":
            decision = "sell"

        confidence = round((sei + (1 - entropy) + bioclock_peak) / 3, 4)

        return {
            "organism": self.organism,
            "decision": decision,
            "confidence": confidence,
            "allow_trade": decision != "hold",
            "trend_direction": trend_direction,
            "criteria": {
                "spiral_phase": phase,
                "sei": round(sei, 4),
                "entropy": round(entropy, 4),
                "z_score": round(z_score, 4),
                "bioclock_peak": round(bioclock_peak, 4)
            }
        }

# Example usage
if __name__ == "__main__":
    sample_executor = SpiralTradeExecutor("EURUSD")

    test_data = {
        "spiral_phase": "rebirth",
        "sei": 0.3413,
        "entropy": 0.2455,
        "z_score": -1.7080,
        "bioclock_peak": 0.8
    }

    result = sample_executor.evaluate_trade_readiness(test_data, trend_direction="buy")
    print("🧬 Spiral Trade Decision:", result)
