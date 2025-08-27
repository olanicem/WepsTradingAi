# ==========================================
# WEPS Escalation Engine
# File: weps/reflex/escalation_engine.py
# Author: Ola (FESI Framework)
# Purpose: Spiral-aware reflex escalation & override system
# ==========================================

import numpy as np
from datetime import datetime
from weps.memory.spiral_memory import load_recent_memory
from weps.rules.life_rules import get_life_rules
from weps.utils.log_utils import log_escalation_event

class EscalationEngine:
    """
    This engine takes reflex decisions and evolves/escalates them
    using mutation memory, trauma history, spiral state, and black swan defense.
    """

    def __init__(self, organism_id):
        self.org_id = organism_id
        self.memory = load_recent_memory(organism_id)
        self.rules = get_life_rules(organism_id)

    def escalate(self, reflex_decision, dna_vector, neuron_outputs, spiral_phase):
        """
        Main escalation logic.
        """
        decision = reflex_decision
        reason = "REFLEX_DIRECT"

        # --- 1. Trauma override ---
        if self._is_in_trauma_zone():
            decision = "SLEEP"
            reason = "TRAUMA_ZONE_OVERRIDE"

        # --- 2. Mutation loop detection ---
        elif self._is_mutation_loop():
            decision = "MUTATE"
            reason = "MUTATION_LOOP_ESCAPE"

        # --- 3. Spiral phase correction ---
        elif self._phase_conflict(spiral_phase, neuron_outputs):
            decision = "HOLD"
            reason = "PHASE_CORRECTION_CONFLICT"

        # --- 4. Immune override ---
        elif neuron_outputs.get("immune_response", 0) > 0.85:
            decision = "SLEEP"
            reason = "BLACK_SWAN_DEFENSE"

        # --- Log result ---
        self._log(decision, reason, dna_vector, neuron_outputs)
        return decision, reason

    def _is_in_trauma_zone(self):
        """
        Check if asset is currently operating in a historical trauma zone.
        """
        trauma_zones = self.memory.get("trauma_zones", [])
        current_price = self.memory.get("latest_price")
        for zone in trauma_zones:
            if zone["low"] <= current_price <= zone["high"]:
                return True
        return False

    def _is_mutation_loop(self):
        """
        Detect if reflex decisions are cycling without resolution.
        """
        past_decisions = self.memory.get("recent_reflexes", [])
        if len(past_decisions) < 5:
            return False
        last_4 = [d["decision"] for d in past_decisions[-4:]]
        return len(set(last_4)) <= 2  # cycling same states

    def _phase_conflict(self, phase, neurons):
        """
        Detect misalignment between phase and active signals.
        E.g., decay phase + high impulse → conflict
        """
        if phase == "DECAY" and neurons.get("impulse", 0) > 0.75:
            return True
        if phase == "REGEN" and neurons.get("metabolic", 0) < 0.3:
            return True
        return False

    def _log(self, decision, reason, dna, neurons):
        """
        Save escalation decision to spiral memory.
        """
        log_escalation_event({
            "organism": self.org_id,
            "decision": decision,
            "reason": reason,
            "dna": dna,
            "neurons": neurons,
            "timestamp": datetime.utcnow().isoformat()
        })
