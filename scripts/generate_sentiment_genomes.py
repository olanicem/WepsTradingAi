#!/usr/bin/env python3
# ====================================================
# 🧬 WEPS Auto Sentiment Genome Generator (Fixed Version)
# Author: Ola Bode (WEPS Creator)
# Description: Generates sentiment genomes in the correct directory
# ====================================================

import os

BASE_GENOME_DIR = os.path.expanduser("~/weps/genome")
SENTIMENT_GENOME_DIR = os.path.join(BASE_GENOME_DIR, "sentiment_genomes")
os.makedirs(SENTIMENT_GENOME_DIR, exist_ok=True)

symbols = [
    f.replace("_genome.py", "").upper()
    for f in os.listdir(BASE_GENOME_DIR)
    if f.endswith("_genome.py") and not f.startswith("__")
]

for symbol in symbols:
    path = os.path.join(SENTIMENT_GENOME_DIR, f"{symbol}_sentiment_genome.py")
    if os.path.exists(path):
        print(f"✅ {symbol} sentiment genome already exists.")
        continue

    with open(path, "w") as f:
        f.write(f'''# {symbol} Sentiment Genome — Auto-Generated
# ===================================================
# 🌐 {symbol} Sentiment Genome — FESI Validated
# Author: Ola Bode (WEPS Creator)
# ===================================================

sentiment_keywords = {{
    "risk_on": ["bullish", "growth", "optimism", "hawkish", "buying pressure"],
    "risk_off": ["bearish", "decline", "fear", "dovish", "selloff", "recession"],
}}

symbol_aliases = ["{symbol}"]

meta = {{
    "default_bias": "neutral",
    "volatility_sensitivity": 0.4,
    "bias_override": {{}},
}}
''')
    print(f"🧬 Generated sentiment genome for {symbol}")
