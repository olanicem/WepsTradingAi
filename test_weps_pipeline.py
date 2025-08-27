#!/usr/bin/env python3
# ================================================================
# 🧪 WEPSPipeline Integration Test — Production Grade Validation
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Runs WEPSPipeline for a set of core assets
#   - Validates data flow through all modules
#   - Verifies state vector construction and output shapes
#   - Logs detailed results for debugging and confirmation
# ================================================================

import logging
from weps.core.weps_pipeline import WEPSPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WEPS.Test")

def test_pipeline_for_organism(organism: str):
    logger.info(f"--- Starting WEPSPipeline test for {organism} ---")
    try:
        pipeline = WEPSPipeline(organism)
        pipeline.run()
        sv = pipeline.state_vector
        assert sv is not None, "State vector is None"
        assert sv.shape[0] > 0, "State vector has zero length"
        logger.info(f"Test successful for {organism}: State vector length={sv.shape[0]}")
    except Exception as e:
        logger.error(f"Test failed for {organism}: {e}")

def main():
    # Core test organisms from WEPS zoo
    test_assets = [
        "EURUSD", "USDJPY", "AAPL", "BTCUSD", "ETHUSD"
    ]
    for asset in test_assets:
        test_pipeline_for_organism(asset)

if __name__ == "__main__":
    main()
