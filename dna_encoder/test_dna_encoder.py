#!/usr/bin/env python3
import logging
from dna_encoder import DNAEncoder
from weps.genome.eurusd_genome import eurusd_genome  # adjust import path if needed

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Fake live data example
    fake_market_data = {
        "atr": 0.008,
        "impulse_strength": 0.010,
        "correction_depth": 0.004,
        "entropy_score": 0.35,
        "mutation_score": 0.002,
        "sentiment_score": 0.25,
        "volume": 1.6e12,
        "correlation_score": 0.7
    }

    encoder = DNAEncoder(eurusd_genome, "EUR/USD", "growth")
    vector, context = encoder.encode(fake_market_data)

    print("\nDNA Vector:", vector)
    print("Context:", context)

