#!/usr/bin/env python3
# ==========================================================
# 🧠 WEPS Reflex Cortex TradeNet
# ✅ Production‑Ready Real-Time Trader
# ==========================================================

import numpy as np
import logging
from typing import Dict, Tuple

from weps.neurons.neuron_processor import NeuronProcessor
from weps.core.reflex_policy_engine import ReflexRLPolicy
from weps.execution.trade_engine import TradeExecutor  # Live trade simulator
from weps.utils.spiral_utils import calculate_exit_strategy
from weps.genome.genome import WEPSGenome
from weps.wave_engine import WEPSWaveEngine
from weps.data_loader import WEPSMasterDataLoader

logger = logging.getLogger("WEPS.ReflexCortex")
logging.basicConfig(level=logging.INFO)

# ==========================================================
# ⚙️ Run Real-Time Reflex Cortex Trade Decision
# ==========================================================
def run_reflex_trade(symbol: str, api_key: str) -> None:
    # ---- 📡 Load Market Data ----
    loader = WEPSMasterDataLoader(api_key=api_key)
    df = loader.fetch_ohlcv(symbol, timescale="1day", limit=300)

    # ---- 🌊 Run Wave + Genome ----
    wave_result = WEPSWaveEngine(df).run()
    genome = WEPSGenome(df, wave_result)

    # ---- 🧠 Extract Final State Vector ----
    state_vector, final_state = NeuronProcessor(df, genome).process()

    # ---- 🤖 RL Reflex Policy Decision ----
    policy = ReflexRLPolicy()
    action, confidence = policy.predict(state_vector)

    # ---- 🎯 Extract Trade Signals ----
    phase_name = final_state.get("phase_transition", {}).get("phase_name", "unknown")
    z_main = final_state.get("z_score_main", 0.0)
    z_sub = final_state.get("z_score_sub", 0.0)
    z_micro = final_state.get("z_score_micro", 0.0)
    timing_factor = final_state.get("corrective_wave_timing_factor", 0.0)

    logger.info(f"🧠 Reflex Decision for [{symbol}] → {action} | Phase: {phase_name} | Confidence: {confidence:.3f}")

    # ---- 📈 Execute Trade if Conditions are Met ----
    if action != "hold" and confidence > 0.6:
        sl, tp, exit_time = calculate_exit_strategy(df, action)
        TradeExecutor.place_trade(symbol=symbol,
                                  action=action,
                                  stop_loss=sl,
                                  take_profit=tp,
                                  exit_time=exit_time,
                                  confidence=confidence,
                                  phase=phase_name,
                                  z_score_main=z_main,
                                  z_score_sub=z_sub,
                                  z_score_micro=z_micro,
                                  timing_factor=timing_factor)
    else:
        logger.info(f"❄️ No trade for [{symbol}]. Holding due to weak signals or confidence.")


# ==========================================================
# 🧪 CLI Entry
# ==========================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run Reflex Cortex Trade Decision")
    parser.add_argument("--symbol", required=True, help="Symbol to trade (e.g., EURUSD)")
    parser.add_argument("--api_key", required=True, help="FMP API key")

    args = parser.parse_args()
    run_reflex_trade(symbol=args.symbol, api_key=args.api_key)
