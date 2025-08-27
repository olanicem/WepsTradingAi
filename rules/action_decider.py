# ============================================================
# WEPS ACTION DECIDER — Spiral Life Reflex Intelligence
# File: weps/rules/action_decider.py
# Author: Ola (WEPS FESI Framework)
# Description: Converts spiral bio-states to intelligent trading actions
# ============================================================

def decide_action_from_biology(spiral_state: dict) -> int:
    """
    Convert a real-time spiral biological state into a trading decision.
    Actions:
        0 = BUY
        1 = SELL
        2 = HOLD
        3 = WAIT
        4 = EXIT
        5 = MUTATE
    """
    phase      = spiral_state.get("phase", "Unknown")
    sei        = float(spiral_state.get("sei", 0.0))
    impulse    = float(spiral_state.get("impulse", 0.0))
    entropy    = float(spiral_state.get("entropy", 0.0))
    trauma     = float(spiral_state.get("memory_shock", 0.0))

    # ---- Core Reflex Logic (Spiral-Aware) ----

    if phase == "Growth" and sei > 0.5 and impulse < 0.6:
        return 0  # BUY

    elif phase in ["Decay", "Death"] and impulse > 0.7:
        return 1  # SELL

    elif phase == "Maturity" and 0.3 <= sei <= 0.6:
        return 2  # HOLD

    elif entropy > 0.8 or 0.5 <= impulse <= 0.6:
        return 3  # WAIT

    elif sei < 0.1 and phase == "Death" and impulse > 0.85:
        return 4  # EXIT

    elif trauma > 0.5 and entropy > 0.6:
        return 5  # MUTATE

    else:
        return 3  # Default to WAIT when unclear
