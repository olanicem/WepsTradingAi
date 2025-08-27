#!/usr/bin/env python3
# ================================================================
# 🧬 WEPS Spiral Mutation Memory v2 — Reflex + Mutation Engine
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Logs mutation, escalation, and spiral phase transitions
#   - Performs biological mutation evaluation based on score delta
#   - Supports recent memory retrieval and decision confirmation
# ================================================================

import os
import json
from datetime import datetime
from typing import Optional, List, Dict

# 🧠 Memory file location
MEMORY_PATH = "weps/memory/spiral_mutation_log.jsonl"

# ⚙️ Mutation thresholds (FESI-tuned)
MUTATION_DELTA_THRESHOLD = 0.12
ESCALATION_TRIGGER_SCORE = 0.85

class SpiralMemory:
    def __init__(self, memory_path: Optional[str] = None):
        self.memory_path = memory_path or MEMORY_PATH
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)

    def log_event(self,
                  symbol: str,
                  phase: str,
                  action: str,
                  score: float,
                  mutation_score: float,
                  escalation_score: Optional[float] = None,
                  meta: Optional[Dict] = None):
        """
        Logs mutation or escalation with timestamp and diagnostics.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol.upper(),
            "phase": phase,
            "action": action,
            "score": round(score, 4),
            "mutation_score": round(mutation_score, 4),
            "escalation_score": round(escalation_score or 0.0, 4),
            "meta": meta or {}
        }

        try:
            with open(self.memory_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            print(f"[SpiralMemory] Logged: {symbol} [{phase}] {action} | Mutation={mutation_score:.3f}")
        except Exception as e:
            print(f"[SpiralMemory] Logging Failed: {e}")

    def load_recent(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Loads recent decisions for a symbol to detect mutation/evolution.
        """
        if not os.path.exists(self.memory_path):
            return []

        try:
            with open(self.memory_path, "r") as f:
                entries = [json.loads(l.strip()) for l in f if l.strip()]
            filtered = [e for e in entries if e.get("symbol") == symbol.upper()]
            return filtered[-limit:]
        except Exception as e:
            print(f"[SpiralMemory] Load Failed: {e}")
            return []

    def detect_mutation(self, symbol: str, current_action: str, current_score: float) -> bool:
        """
        Detects whether a mutation (i.e., change of state) occurred compared to prior decision.
        """
        recent = self.load_recent(symbol, limit=2)
        if len(recent) < 1:
            return False

        prev = recent[-1]
        prev_action = prev.get("action")
        prev_score = prev.get("score", 0.0)

        action_changed = prev_action != current_action
        score_delta = abs(current_score - prev_score)

        if action_changed and score_delta > MUTATION_DELTA_THRESHOLD:
            print(f"[MutationDetected] {symbol}: {prev_action} → {current_action}, Δ={score_delta:.3f}")
            return True
        return False

    def should_escalate(self, score: float, mutation_score: float) -> bool:
        """
        Determines if decision qualifies for escalation based on biological confidence and mutation risk.
        """
        return score >= ESCALATION_TRIGGER_SCORE and mutation_score >= 0.2
