#!/usr/bin/env python3
# ================================================================
# 📜 WEPS LogUtils v2.0 — Spiral Intelligence Logging System
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Logs key spiral events: phase transitions, decision intelligence, trade history
#   - Real-time JSON printouts and persistent logging
#   - FESI Compliant
# ================================================================

import os
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("WEPS.LogUtils")

# Paths
LOG_DIR = Path.home() / "weps" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PHASE_LOG_PATH = LOG_DIR / "spiral_phase_log.jsonl"
DECISION_LOG_PATH = LOG_DIR / "reflex_decision_log.jsonl"
TRADE_SUMMARY_LOG_PATH = LOG_DIR / "trade_summary_log.jsonl"

# ------------------------------------------------------------
# 🌪️ Phase Logging
# ------------------------------------------------------------

def log_spiral_phase(symbol: str, context: dict, entropy: float):
    phase = context.get("phase", "unknown")
    sentiment = context.get("sentiment", {}).get("sentiment_signal", "neutral")
    sei = round(context.get("sei", 0.0), 4)
    zscore = round(context.get("zscore", 0.0), 2)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "phase": phase,
        "entropy": round(entropy, 4),
        "sei": sei,
        "zscore": zscore,
        "sentiment": sentiment
    }

    _append_to_log(PHASE_LOG_PATH, record)

    # Print JSON to terminal
    print(json.dumps({
        "🔬 SpiralPhase": phase,
        "📉 Entropy": entropy,
        "📈 SEI": sei,
        "💬 Sentiment": sentiment,
        "🧪 Z": zscore,
        "🧬 Symbol": symbol
    }, indent=2))


# ------------------------------------------------------------
# 🧠 Reflex Decision Logging
# ------------------------------------------------------------

def log_spiral_decision(symbol: str, decision: dict):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "phase": decision.get("phase"),
        "action": decision.get("action"),
        "entry": decision.get("entry"),
        "sl": decision.get("sl"),
        "tp": decision.get("tp"),
        "confidence": decision.get("confidence"),
        "mutation_score": decision.get("mutation_score"),
        "escalation_score": decision.get("escalation_score"),
        "support_genes": decision.get("support_genes", []),
        "reject_genes": decision.get("reject_genes", [])
    }

    _append_to_log(DECISION_LOG_PATH, record)

    print(json.dumps({
        "🧠 Decision": decision.get("action"),
        "📍 Entry": decision.get("entry"),
        "🛡️ SL": decision.get("sl"),
        "🎯 TP": decision.get("tp"),
        "📊 Confidence": decision.get("confidence"),
        "🧬 Phase": decision.get("phase"),
        "🧪 Mutation": decision.get("mutation_score"),
        "📈 Escalation": decision.get("escalation_score")
    }, indent=2))


# ------------------------------------------------------------
# 📊 Trade Summary Logging
# ------------------------------------------------------------

def log_trade_summary(entry: dict):
    _append_to_log(TRADE_SUMMARY_LOG_PATH, entry)

    print(json.dumps({
        "💹 Trade Summary": {
            "Symbol": entry.get("symbol"),
            "Action": entry.get("action"),
            "Executed Entry": entry.get("entry"),
            "Requested Entry": entry.get("entry_requested"),
            "SL": entry.get("sl"),
            "TP": entry.get("tp"),
            "Phase": entry.get("phase"),
            "Confidence": entry.get("confidence"),
            "Slippage BPS": entry.get("slippage_bps"),
        }
    }, indent=2))


# ------------------------------------------------------------
# 🧾 Append Helper
# ------------------------------------------------------------

def _append_to_log(path: Path, data: dict):
    try:
        with open(path, "a") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        logger.warning(f"[LogUtils] Failed to append to {path.name}: {e}")
