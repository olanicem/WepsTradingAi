#!/usr/bin/env python3
# ================================================================
# 🧬 WEPS Supreme Spiral Live Orchestrator v6.5 — FESI Compliant
# Author: Ola Bode — Final Reflex Execution Engine
# Description:
#   - Real-time phase+entropy+reflex monitoring and decision execution
#   - JSON printouts of spiral intelligence (candle, wave, entropy, mutation)
#   - Mutation logging, Telegram alerts, trade logging
# ================================================================

import os
import time
import json
import logging
from datetime import datetime
import numpy as np

from weps.utils.live_data_feeder import LiveDataFeeder
from weps.utils.live_data_polling import poll_latest_ohlcv
from weps.core.reflex_policy_engine import ReflexPolicyEngine
from weps.memory.spiral_memory import SpiralMemory
from weps.live.oanda_trade_executor import execute_oanda_trade
from weps.utils.telegram_bot import send_trade_alert
from weps.utils.log_utils import (
    log_spiral_phase, log_spiral_decision, log_trade_summary
)

# ⚙️ Config
TIMEFRAMES = ["1h", "4h", "1d"]
PREFERRED_PRICE_TF = "1h"
POLL_INTERVAL_SEC = 300
BASE_LOT = 1.0
GENOME_PATH = os.path.expanduser("~/weps/genome")

logger = logging.getLogger("WEPS.LiveOrchestrator")
logging.basicConfig(level=logging.INFO)
memory = SpiralMemory()

def get_all_genome_organisms():
    return [
        f.replace("_genome.py", "").upper()
        for f in os.listdir(GENOME_PATH)
        if f.endswith("_genome.py") and not f.startswith("__") and "sentiment" not in f
    ]

def dynamic_lot(confidence: float, volatility: float) -> float:
    risk_factor = max((confidence * volatility) ** 0.5, 0.25)
    return round(BASE_LOT * min(risk_factor, 2.0), 2)

def validate_state_vector(symbol: str, vector: dict) -> bool:
    required_keys = ["impulse", "volatility", "trend", "sentiment"]
    if not isinstance(vector, dict):
        logger.warning(f"[{symbol}] State vector is not a dict.")
        return False
    missing = [k for k in required_keys if k not in vector or not isinstance(vector[k], dict)]
    if missing:
        logger.warning(f"[{symbol}] State vector missing/invalid: {missing}")
        return False
    return True

def run_live_orchestrator():
    logger.info("🧬 WEPS Supreme Spiral Live Orchestrator v6.5 Starting...")
    organisms = get_all_genome_organisms()
    logger.info(f"🔍 Detected Organisms: {organisms}")

    feeders = {s: LiveDataFeeder(s, TIMEFRAMES, PREFERRED_PRICE_TF) for s in organisms}

    while True:
        for symbol in organisms:
            try:
                logger.info(f"🔁 [{symbol}] Fetching latest OHLCV...")
                fresh_dfs = {tf: poll_latest_ohlcv(symbol, tf) for tf in TIMEFRAMES}
                if any(df is None or df.empty for df in fresh_dfs.values()):
                    logger.warning(f"[{symbol}] OHLCV incomplete. Skipping.")
                    continue

                feeder = feeders[symbol]
                payload = feeder.build_live_payload(symbol, fresh_dfs)

                raw_vector = payload.get("state_vector")
                context = payload.get("context", {})

                # Fix ndarray
                if isinstance(raw_vector, np.ndarray):
                    keys = context.get("vector_keys", [])
                    raw_vector = dict(zip(keys, raw_vector))
                    logger.warning(f"[{symbol}] Auto-healed ndarray state_vector.")

                if not validate_state_vector(symbol, raw_vector):
                    logger.warning(f"[{symbol}] Invalid vector. Skipping.")
                    continue

                # 🔬 Print spiral status
                candle_result = context.get("neurons", {}).get("candle_pattern", {}).get("pattern", "unknown")
                wave = context.get("wave_result", {}).get("type", "none")

                print(json.dumps({
                    "🧬": symbol,
                    "🌊 Phase": context.get("phase"),
                    "🕯️ CandlePattern": candle_result,
                    "🌪️ Wave": wave,
                    "📉 Entropy": round(context.get("entropy", 0.0), 4),
                    "📈 SEI": round(context.get("sei", 0.0), 4)
                }, indent=2))

                # 📜 Log Phase
                log_spiral_phase(symbol, context, context.get("entropy", 0.0))

                # 🧠 Reflex Cortex
                cortex = ReflexPolicyEngine(
                    organism=symbol,
                    spiral_vector=raw_vector,
                    wave_result=context.get("wave_result", {}),
                    df=context.get("raw_df"),
                    entropy_slope=context.get("entropy"),
                    neuron_processor=context.get("neurons"),
                    spiral_phase=context.get("phase"),
                    x=context.get("timing_score"),
                    sentiment_output=context.get("sentiment")
                )

                decision = cortex.decide()
                if not isinstance(decision, dict) or "action" not in decision:
                    logger.warning(f"[{symbol}] Invalid decision format.")
                    continue

                log_spiral_decision(symbol, decision)

                action = decision.get("action", "hold")
                confidence = float(decision.get("confidence", 0.0))
                volatility = float(context.get("volatility", 0.01))
                lot = dynamic_lot(confidence, volatility)

                if action in ["buy", "sell"]:
                    logger.info(f"📈 [{symbol}] {action.upper()} | Phase={decision.get('phase')} | Confidence={confidence:.2f} | Lot={lot}")

                    execution = execute_oanda_trade(
                        symbol=symbol,
                        action=action,
                        lot_size=lot,
                        sl=decision["sl"],
                        tp=decision["tp"]
                    )

                    entry_exec = execution.get("executed_price", decision.get("entry"))
                    slippage = round((abs(entry_exec - decision["entry"]) / decision["entry"]) * 10000, 2)

                    memory.log_mutation(
                        symbol=symbol,
                        phase=decision.get("phase"),
                        action=action,
                        score=confidence,
                        mutation_score=decision.get("mutation_score", 0.0),
                        escalation_score=decision.get("escalation_score", 0.0),
                        timestamp=datetime.utcnow().isoformat()
                    )

                    send_trade_alert(
                        symbol=symbol,
                        action=action,
                        confidence=confidence,
                        phase=decision.get("phase"),
                        sl=decision["sl"],
                        tp=decision["tp"],
                        slippage_bps=slippage
                    )

                    log_trade_summary({
                        "timestamp": datetime.utcnow().isoformat(),
                        "symbol": symbol,
                        "action": action,
                        "entry": entry_exec,
                        "entry_requested": decision.get("entry"),
                        "sl": decision["sl"],
                        "tp": decision["tp"],
                        "phase": decision.get("phase"),
                        "confidence": confidence,
                        "mutation_score": decision.get("mutation_score"),
                        "escalation_score": decision.get("escalation_score"),
                        "support_genes": decision.get("support_genes", []),
                        "reject_genes": decision.get("reject_genes", []),
                        "slippage_bps": slippage
                    })

                else:
                    logger.info(f"⏸️ [{symbol}] HOLD | Phase={decision.get('phase')} | Confidence={confidence:.2f}")

            except Exception as e:
                logger.error(f"⚠️ [{symbol}] Exception: {e}", exc_info=True)

        logger.info(f"🕒 Sleeping {POLL_INTERVAL_SEC}s...\n")
        time.sleep(POLL_INTERVAL_SEC)

if __name__ == "__main__":
    run_live_orchestrator()
