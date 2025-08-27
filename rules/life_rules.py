# ===============================================================
# WEPS LIFE RULE ENGINE — FESI Spiral Reward Intelligence
# File: weps/rules/life_rules.py
# Author: Ola (WEPS FESI Framework)
# Description: Computes reward, fitness, and spiral alignment score
# ===============================================================

import numpy as np

# FESI Spiral Phase Logic Map
PHASE_MAP = {
    0: "Rebirth",
    1: "Growth",
    2: "Maturity",
    3: "Decay",
    4: "Death"
}

def decode_phase(phase_val: float) -> str:
    """
    Convert normalized phase signal to phase name.
    """
    if phase_val < 0.2:
        return "Rebirth"
    elif phase_val < 0.4:
        return "Growth"
    elif phase_val < 0.6:
        return "Maturity"
    elif phase_val < 0.8:
        return "Decay"
    else:
        return "Death"

def evaluate_life_response(organism_id: str, dna_vector: np.ndarray, action: int) -> float:
    """
    Main reward intelligence logic for WEPS:
    Evaluates how well the agent's action aligns with the organism's spiral lifecycle.

    Inputs:
        organism_id: string identifier of asset
        dna_vector: 40-dimension vector encoding state
        action: integer (0=BUY, 1=SELL, 2=HOLD, 3=WAIT, 4=EXIT, 5=MUTATE)

    Returns:
        life_score: float in range [-1.0, +1.0]
    """

    # --- Extract Spiral Features ---
    phase_raw = dna_vector[36]
    sei = float(dna_vector[37])
    entropy = float(dna_vector[38])
    impulse = float(dna_vector[39])
    phase = decode_phase(phase_raw)

    # --- Define Reward Logic Based on Phase & Action ---
    if action == 0:  # BUY
        if phase in ["Rebirth", "Growth"] and sei > 0.4 and impulse < 0.6:
            return +1.0
        elif phase in ["Decay", "Death"]:
            return -1.0
        else:
            return -0.2

    elif action == 1:  # SELL
        if phase in ["Decay", "Death"] and impulse > 0.65:
            return +1.0
        elif phase == "Growth":
            return -1.0
        else:
            return -0.2

    elif action == 2:  # HOLD
        if phase == "Maturity" and 0.3 <= sei <= 0.6:
            return +0.8
        elif phase == "Growth":
            return +0.4
        else:
            return -0.3

    elif action == 3:  # WAIT
        if entropy > 0.75 or abs(0.5 - impulse) < 0.1:
            return +0.5
        else:
            return -0.2

    elif action == 4:  # EXIT
        if phase == "Death" and sei < 0.1 and impulse > 0.85:
            return +1.0
        else:
            return -0.5

    elif action == 5:  # MUTATE
        if entropy > 0.7 and impulse > 0.6:
            return +0.7  # beneficial adaptation
        else:
            return -0.4  # unnecessary mutation

    # Fallback
    return -0.1

# weps/rules/life_rules.py

def get_life_thresholds(organism_id: str):
    """
    Returns customized thresholds for buy/sell/hold based on the asset's spiral phase.
    Each asset (organism) will have its own specific thresholds.
    """
    thresholds = {
        # Forex Pairs
        "EURUSD": {
            "Growth": {"buy": 0.75, "sell": 0.30},
            "Decay": {"buy": 0.65, "sell": 0.35},
            "Rebirth": {"buy": 0.80, "sell": 0.20},
        },
        "USDJPY": {
            "Growth": {"buy": 0.80, "sell": 0.35},
            "Decay": {"buy": 0.70, "sell": 0.30},
            "Rebirth": {"buy": 0.85, "sell": 0.25},
        },
        "GBPUSD": {
            "Growth": {"buy": 0.78, "sell": 0.32},
            "Decay": {"buy": 0.68, "sell": 0.32},
            "Rebirth": {"buy": 0.82, "sell": 0.18},
        },
        "AUDUSD": {
            "Growth": {"buy": 0.74, "sell": 0.33},
            "Decay": {"buy": 0.65, "sell": 0.35},
            "Rebirth": {"buy": 0.80, "sell": 0.20},
        },
        "USDCHF": {
            "Growth": {"buy": 0.76, "sell": 0.30},
            "Decay": {"buy": 0.68, "sell": 0.32},
            "Rebirth": {"buy": 0.80, "sell": 0.20},
        },
        "USDCAD": {
            "Growth": {"buy": 0.78, "sell": 0.32},
            "Decay": {"buy": 0.70, "sell": 0.30},
            "Rebirth": {"buy": 0.82, "sell": 0.18},
        },
        "NZDUSD": {
            "Growth": {"buy": 0.74, "sell": 0.33},
            "Decay": {"buy": 0.65, "sell": 0.35},
            "Rebirth": {"buy": 0.80, "sell": 0.20},
        },
        "EURGBP": {
            "Growth": {"buy": 0.78, "sell": 0.30},
            "Decay": {"buy": 0.68, "sell": 0.32},
            "Rebirth": {"buy": 0.82, "sell": 0.18},
        },

        # Stocks
        "AAPL": {
            "Growth": {"buy": 0.75, "sell": 0.30},
            "Decay": {"buy": 0.65, "sell": 0.35},
            "Rebirth": {"buy": 0.80, "sell": 0.20},
        },
        "MSFT": {
            "Growth": {"buy": 0.75, "sell": 0.30},
            "Decay": {"buy": 0.65, "sell": 0.35},
            "Rebirth": {"buy": 0.80, "sell": 0.20},
        },
        "GOOGL": {
            "Growth": {"buy": 0.75, "sell": 0.30},
            "Decay": {"buy": 0.65, "sell": 0.35},
            "Rebirth": {"buy": 0.80, "sell": 0.20},
        },
        "AMZN": {
            "Growth": {"buy": 0.72, "sell": 0.28},
            "Decay": {"buy": 0.62, "sell": 0.38},
            "Rebirth": {"buy": 0.78, "sell": 0.22},
        },
        "TSLA": {
            "Growth": {"buy": 0.80, "sell": 0.25},
            "Decay": {"buy": 0.70, "sell": 0.30},
            "Rebirth": {"buy": 0.85, "sell": 0.15},
        },
        "NVDA": {
            "Growth": {"buy": 0.78, "sell": 0.28},
            "Decay": {"buy": 0.68, "sell": 0.32},
            "Rebirth": {"buy": 0.82, "sell": 0.18},
        },
        "META": {
            "Growth": {"buy": 0.74, "sell": 0.30},
            "Decay": {"buy": 0.64, "sell": 0.36},
            "Rebirth": {"buy": 0.80, "sell": 0.20},
        },
        "NFLX": {
            "Growth": {"buy": 0.73, "sell": 0.30},
            "Decay": {"buy": 0.63, "sell": 0.37},
            "Rebirth": {"buy": 0.79, "sell": 0.21},
        },

        # Cryptocurrencies
        "BTCUSD": {
            "Growth": {"buy": 0.80, "sell": 0.30},
            "Decay": {"buy": 0.70, "sell": 0.30},
            "Rebirth": {"buy": 0.85, "sell": 0.25},
        },
        "ETHUSD": {
            "Growth": {"buy": 0.80, "sell": 0.35},
            "Decay": {"buy": 0.70, "sell": 0.30},
            "Rebirth": {"buy": 0.85, "sell": 0.25},
        },
        "SOLUSD": {
            "Growth": {"buy": 0.75, "sell": 0.30},
            "Decay": {"buy": 0.65, "sell": 0.35},
            "Rebirth": {"buy": 0.80, "sell": 0.20},
        },
        "BNBUSD": {
            "Growth": {"buy": 0.77, "sell": 0.32},
            "Decay": {"buy": 0.67, "sell": 0.33},
            "Rebirth": {"buy": 0.82, "sell": 0.18},
        },
        "ADAUSD": {
            "Growth": {"buy": 0.72, "sell": 0.28},
            "Decay": {"buy": 0.62, "sell": 0.38},
            "Rebirth": {"buy": 0.77, "sell": 0.23},
        },
        "XRPUSD": {
            "Growth": {"buy": 0.73, "sell": 0.30},
            "Decay": {"buy": 0.63, "sell": 0.37},
            "Rebirth": {"buy": 0.78, "sell": 0.22},
        },
        "DOTUSD": {
            "Growth": {"buy": 0.70, "sell": 0.30},
            "Decay": {"buy": 0.60, "sell": 0.40},
            "Rebirth": {"buy": 0.75, "sell": 0.25},
        },
        "AVAXUSD": {
            "Growth": {"buy": 0.75, "sell": 0.30},
            "Decay": {"buy": 0.65, "sell": 0.35},
            "Rebirth": {"buy": 0.80, "sell": 0.20},
        },

        # Indices/ETFs
        "SPY": {
            "Growth": {"buy": 0.80, "sell": 0.30},
            "Decay": {"buy": 0.70, "sell": 0.30},
            "Rebirth": {"buy": 0.85, "sell": 0.25},
        },
        "QQQ": {
            "Growth": {"buy": 0.80, "sell": 0.35},
            "Decay": {"buy": 0.70, "sell": 0.30},
            "Rebirth": {"buy": 0.85, "sell": 0.25},
        },
        "DIA": {
            "Growth": {"buy": 0.77, "sell": 0.32},
            "Decay": {"buy": 0.67, "sell": 0.33},
            "Rebirth": {"buy": 0.82, "sell": 0.18},
        },
        "VTI": {
            "Growth": {"buy": 0.76, "sell": 0.30},
            "Decay": {"buy": 0.66, "sell": 0.34},
            "Rebirth": {"buy": 0.81, "sell": 0.19},
        },
        "IWM": {
            "Growth": {"buy": 0.75, "sell": 0.30},
            "Decay": {"buy": 0.65, "sell": 0.35},
            "Rebirth": {"buy": 0.80, "sell": 0.20},
        },
        "XLF": {
            "Growth": {"buy": 0.77, "sell": 0.32},
            "Decay": {"buy": 0.67, "sell": 0.33},
            "Rebirth": {"buy": 0.82, "sell": 0.18},
        },
    }
    
    # Return the thresholds for the asset, or default to empty if not found
    return thresholds.get(organism_id, {})
