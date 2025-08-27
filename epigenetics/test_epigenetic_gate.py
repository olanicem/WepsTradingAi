#!/usr/bin/env python3
# ==========================================================
# 🧪 WEPS EpigeneticGate Test Script
# ✅ Tests gene activation & vector adjustment
# ==========================================================
import numpy as np
import logging
from epigenetic_gate import EpigeneticGate

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Example DNA vector & context
    test_vector = np.array([0.2, 0.5, 0.3, 0.4, 0.1, 0.6, 0.7, 0.8], dtype=np.float32)
    test_context = {
        "entropy_norm": 0.65,
        "mutation_norm": 0.35,
        "sentiment_norm": 0.25
    }

    # Initialize EpigeneticGate for 'rebirth' phase
    gate = EpigeneticGate(phase="rebirth")
    active_genes, adjusted_vector = gate.evaluate(test_vector, test_context)

    print("\n🎯 Active Genes:", active_genes)
    print("🧬 Adjusted DNA Vector:", adjusted_vector)
