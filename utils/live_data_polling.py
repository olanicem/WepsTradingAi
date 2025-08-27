#!/usr/bin/env python3
# ================================================================
# 🧬 WEPS LiveDataPolling — Spiral-Aligned Real-Time Polling Module
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Periodically fetches incremental OHLCV updates from FMP
#   - Updates LiveDataFeeder and Spiral Pipeline in place
#   - Provides fast-access polling function for Orchestrator mode
# ================================================================

import time
import logging
from typing import Dict, Optional
import pandas as pd

from weps.data_loader import WEPSMasterDataLoader
from weps.utils.live_data_updater import LiveDataUpdater
from weps.utils.live_data_feeder import LiveDataFeeder

logger = logging.getLogger("WEPS.LiveDataPolling")
logger.setLevel(logging.INFO)

# ================================================================
# 🌐 Live Polling Class — Streaming Live Data to Feeder
# ================================================================

class LiveDataPolling:
    def __init__(
        self,
        organism: str,
        live_feeder: LiveDataFeeder,
        polling_interval_sec: int = 60,
        max_candles_fetch: int = 100,
        timeframes: Optional[list] = None,
        api_key: Optional[str] = None,
    ):
        self.organism = organism.upper()
        self.polling_interval_sec = polling_interval_sec
        self.max_candles_fetch = max_candles_fetch
        self.timeframes = timeframes or ["1h", "4h", "1d"]

        self.data_loader = WEPSMasterDataLoader(api_key=api_key)
        self.live_feeder = live_feeder
        self.updater = LiveDataUpdater(live_feeder.live_updater.dfs if live_feeder else None)

    def poll_and_update(self):
        """
        🔁 Poll FMP for latest candles and update the Live Feeder pipeline incrementally.
        """
        logger.info(f"[{self.organism}] Polling FMP for new OHLCV candles...")

        try:
            fetched_data = self.data_loader.fetch_multi_timeframe_ohlcv(
                self.organism,
                limit=self.max_candles_fetch,
            )
            incremental_updates = {}

            for tf in self.timeframes:
                new_df = fetched_data.get(tf)
                if new_df is None or new_df.empty:
                    logger.warning(f"[{self.organism}] No data fetched for {tf}.")
                    continue

                existing_df = self.updater.dfs.get(tf)
                if existing_df is not None and not existing_df.empty:
                    last_known_date = existing_df['date'].iloc[-1]
                    new_candles = new_df[new_df['date'] > last_known_date]
                else:
                    new_candles = new_df

                if not new_candles.empty:
                    incremental_updates[tf] = new_candles
                    logger.info(f"[{self.organism}] {len(new_candles)} new candles for {tf}")

            if incremental_updates:
                for tf, df_new in incremental_updates.items():
                    self.updater.update_timeframe(tf, df_new)

                # Sync into Live Feeder
                self.live_feeder.live_updater.dfs = self.updater.dfs
                self.live_feeder.pipeline_interface.dfs = self.updater.get_data_slices(
                    max(len(df) for df in self.updater.dfs.values()) - 1
                )
                self.live_feeder.pipeline_interface.pipeline.dfs = self.live_feeder.pipeline_interface.dfs
                logger.info(f"[{self.organism}] ✅ Live feeder updated with latest candles.")
            else:
                logger.info(f"[{self.organism}] ⏸️ No new candles detected.")

        except Exception as e:
            logger.error(f"[{self.organism}] ❌ Error during polling: {e}", exc_info=True)

    def start_polling_loop(self):
        """
        🔁 Persistent polling loop — suitable for background daemon execution.
        """
        logger.info(f"[{self.organism}] 🌀 Starting polling loop every {self.polling_interval_sec}s...")
        try:
            while True:
                self.poll_and_update()
                time.sleep(self.polling_interval_sec)
        except KeyboardInterrupt:
            logger.info(f"[{self.organism}] ⛔ Polling loop interrupted by user.")


# ================================================================
# 🧠 Spiral AI External Poller — Used by Live Orchestrator
# ================================================================

def poll_latest_ohlcv(symbol: str, timeframe: str = "1h", api_key: Optional[str] = None) -> pd.DataFrame:
    """
    🚨 External polling shortcut for WEPS Live Orchestrator.
    Returns the freshest OHLCV DataFrame for a given symbol + timeframe.
    """
    try:
        loader = WEPSMasterDataLoader(api_key=api_key)
        data = loader.fetch_multi_timeframe_ohlcv(symbol.upper(), limit=100)
        df = data.get(timeframe)
        if df is None or df.empty:
            raise ValueError(f"No OHLCV data returned for {symbol} [{timeframe}]")
        return df
    except Exception as e:
        logger.error(f"[poll_latest_ohlcv] ❌ Failed for {symbol} [{timeframe}]: {e}")
        raise e
