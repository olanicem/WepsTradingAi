#!/usr/bin/env python3
# ==========================================================
# 🧬 WEPS Genome Hub — Finalized Universal Genome Loader
# ✅ Central Gateway to 20+ WEPS Organisms
# ✅ Supports Full Genome & Dynamic Sentiment Genome
# ✅ FESI-Compliant, Robust Error Handling, Traceable Audit Logs
# Author: Ola Bode (WEPS Creator)
# ==========================================================

import os
import logging
import importlib.util

# === Import full organism genomes ===
from weps.genome import (
    eurusd_genome, usdjpy_genome, gbpusd_genome, eurgbp_genome,
    btcusd_genome, ethusd_genome, dogeusd_genome, adausd_genome,
    aapl_genome, msft_genome, tsla_genome, meta_genome,
    eurjpy_genome, audusd_genome, usdcad_genome, usdchf_genome,
    nzdusd_genome
)

logger = logging.getLogger("WEPS.GenomeHub")

# 📚 Central registry of full genome definitions
GENOME_REGISTRY = {
    "EURUSD": eurusd_genome.GENOME,
    "USDJPY": usdjpy_genome.GENOME,
    "GBPUSD": gbpusd_genome.GENOME,
    "EURGBP": eurgbp_genome.GENOME,
    "BTCUSD": btcusd_genome.GENOME,
    "ETHUSD": ethusd_genome.GENOME,
    "DOGEUSD": dogeusd_genome.GENOME,
    "ADAUSD": adausd_genome.GENOME,
    "AAPL": aapl_genome.GENOME,
    "MSFT": msft_genome.GENOME,
    "TSLA": tsla_genome.GENOME,
    "META": meta_genome.GENOME,
    "EURJPY": eurjpy_genome.GENOME,
    "AUDUSD": audusd_genome.GENOME,
    "USDCAD": usdcad_genome.GENOME,
    "USDCHF": usdchf_genome.GENOME,
    "NZDUSD": nzdusd_genome.GENOME,
}

# ==========================================================
# 🔬 Genome Loader
# ==========================================================

def load_genome(symbol: str) -> dict:
    """
    Loads the full WEPS genome for a given organism.

    Args:
        symbol (str): Organism symbol (e.g., "EURUSD").

    Returns:
        dict: Full genome structure for the organism.

    Raises:
        KeyError: If organism is not found in registry.
    """
    key = symbol.upper()
    if key not in GENOME_REGISTRY:
        logger.error("[GenomeHub] ❌ Unknown organism symbol: %s", symbol)
        raise KeyError(f"[GenomeHub] ❌ Unknown organism symbol: {symbol}")

    logger.info("[GenomeHub] ✅ Loaded genome for organism: %s", symbol)
    return GENOME_REGISTRY[key]

# ==========================================================
# 🧠 Dynamic Sentiment Genome Loader
# ==========================================================

def load_sentiment_genome(symbol: str) -> dict:
    """
    Dynamically imports the sentiment genome for the given symbol
    from ~/weps/genome/sentiment_genomes/.

    Returns:
        dict: {
            "sentiment_keywords": dict,
            "symbol_aliases": list,
            "meta": dict
        }

    Raises:
        FileNotFoundError: If sentiment genome is not found.
        ImportError: If sentiment genome cannot be loaded.
    """
    filename = f"{symbol.upper()}_sentiment_genome.py"
    path = os.path.expanduser(f"~/weps/genome/sentiment_genomes/{filename}")

    if not os.path.exists(path):
        logger.warning("[GenomeHub] ⚠️ Sentiment genome not found for %s", symbol)
        raise FileNotFoundError(f"Sentiment genome not found for {symbol}")

    module_name = f"{symbol.lower()}_sentiment_genome"
    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"[GenomeHub] ❌ Could not load sentiment genome module for {symbol}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "sentiment_keywords"):
        raise KeyError(f"[GenomeHub] ❌ 'sentiment_keywords' not defined in {filename}")

    logger.info("[GenomeHub] 🧠 Loaded sentiment genome for %s", symbol)

    return {
        "sentiment_keywords": module.sentiment_keywords,
        "symbol_aliases": getattr(module, "symbol_aliases", [symbol]),
        "meta": getattr(module, "meta", {})
    }
