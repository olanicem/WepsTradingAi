#!/usr/bin/env python3
# ==========================================================
# 🧪 SentimentNeuron Test — Genome + Live Fetch Diagnostic
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import logging
from pprint import pprint
from weps.neurons.sentiment_neuron import SentimentNeuron

# Setup logging
logging.basicConfig(level=logging.INFO)

# Choose a valid organism symbol with genome (e.g., EURUSD, AAPL)
organism = "EURUSD"

# Initialize neuron
neuron = SentimentNeuron(organism)

# Run the compute() method
result = neuron.compute(
    spiral_meta={"state_name": "growth"},
    wave_result={"direction": "up"},
    current_phase="growth"
)

# Output the result
print(f"\n🧠 SentimentNeuron Result for [{organism}]:")
pprint(result)
