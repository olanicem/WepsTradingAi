#!/usr/bin/env python3
# ==========================================================
# 🧠 WEPS SentimentNeuron — Reflex Cortex FESI Neuron
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Uses NewsSentimentFetcher to assess risk-on/risk-off signals
#   - Aligns sentiment to Spiral Phase and Wave Direction
#   - Provides reflex-aware sentiment adjustment
# ==========================================================

import logging
from typing import Dict, Any, Optional
from weps.neurons.news_sentiment_fetcher import NewsSentimentFetcher

logger = logging.getLogger("WEPS.Neurons.SentimentNeuron")

class SentimentNeuron:
    """
    A real-time FESI reflex neuron for sentiment interpretation.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        try:
            self.fetcher = NewsSentimentFetcher(symbol=self.symbol)
            logger.info(f"[SentimentNeuron] ✅ Initialized for symbol={self.symbol}")
        except Exception as e:
            logger.error(f"[SentimentNeuron] ❌ Initialization failed: {e}", exc_info=True)
            self.fetcher = None

    def compute(
        self,
        df: Optional[Any] = None,
        spiral_meta: Optional[Dict[str, Any]] = None,
        wave_result: Optional[Dict[str, Any]] = None,
        current_phase: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Computes the sentiment signal with fallback logic.

        Args:
            df: Raw OHLCV DataFrame (not used, passed for uniformity)
            spiral_meta: Spiral context, expected to include state_name
            wave_result: Wave engine output, expected to include direction
            current_phase: Direct phase input override
            kwargs: Future-proof arguments

        Returns:
            Dict[str, Any]: {
                sentiment_signal,
                sentiment_adjustment,
                average_polarity,
                phase,
                elliot_direction,
                fallback
            }
        """
        elliot_direction = "unknown"
        phase = current_phase or spiral_meta.get("state_name", "unknown") if spiral_meta else "unknown"

        if isinstance(wave_result, dict):
            elliot_direction = wave_result.get("direction", "unknown")

        if not self.fetcher:
            logger.warning("[SentimentNeuron] ❗️Fetcher unavailable — fallback mode engaged.")
            return self._fallback(phase, elliot_direction)

        try:
            result = self.fetcher.run(
                elliot_direction=elliot_direction,
                phase=phase
            )

            if not isinstance(result, dict):
                logger.warning("[SentimentNeuron] Invalid fetcher output, using fallback.")
                return self._fallback(phase, elliot_direction)

            # Confirm structure
            required = {"sentiment_signal", "sentiment_adjustment", "average_polarity"}
            if not required.issubset(result.keys()):
                logger.warning("[SentimentNeuron] Output missing required keys, using fallback.")
                return self._fallback(phase, elliot_direction)

            result.update({
                "phase": phase,
                "elliot_direction": elliot_direction,
                "fallback": False,
            })

            logger.info(f"[SentimentNeuron] ✅ Computed: {result}")
            return result

        except Exception as e:
            logger.error(f"[SentimentNeuron] ❌ Exception in compute(): {e}", exc_info=True)
            return self._fallback(phase, elliot_direction)

    def _fallback(self, phase: str, direction: str) -> Dict[str, Any]:
        """
        Provides a neutral fallback sentiment state.
        """
        return {
            "sentiment_signal": "neutral",
            "sentiment_adjustment": 0.0,
            "average_polarity": 0.0,
            "phase": phase or "unknown",
            "elliot_direction": direction or "unknown",
            "fallback": True,
        }
