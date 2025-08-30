#!/usr/bin/env python3
# ==========================================================
# 🌊 WEPS SpiralWaveEngineV2.3 — Rule-Based Phase Annotation
# Fully FESI Compliant, Production-Ready
# ==========================================================

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import talib
from scipy.signal import find_peaks
from weps.utils.candle_library import JapaneseCandleLibrary
from weps.utils.log_utils import setup_logger
from weps.utils.config import load_config

logger = setup_logger(__name__, level=logging.INFO)
config = load_config("config/weps_config.yaml")

PHASES = ["rebirth", "growth", "decay", "death"]

def clean_for_json(obj: Any) -> Any:
    """Ensure JSON-serializable output."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj

class SwingDetector:
    """Detect swing highs and lows in OHLCV data."""

    def __init__(self, df: pd.DataFrame, asset_class: str) -> None:
        self.df = df
        self.asset_class = asset_class

    def detect_swings(self) -> List[Tuple[int, float]]:
        """Detect swing points using peak detection."""
        closes = self.df["close"].values
        if len(closes) < 30:
            logger.warning("Insufficient data for swing detection")
            return []

        distance = config.get("swing_distance", 5)
        peaks, _ = find_peaks(closes, distance=distance)
        troughs, _ = find_peaks(-closes, distance=distance)
        indices = np.sort(np.concatenate((peaks, troughs)))
        swings = [(i, closes[i]) for i in indices if i < len(closes)]

        return self._filter_swings(swings)

    def _filter_swings(self, swings: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Filter insignificant swings based on ATR and entropy."""
        if len(swings) < 2:
            return swings

        closes = [p for _, p in swings]
        diffs = np.abs(np.diff(closes))
        atr = np.mean(diffs[-30:]) if len(diffs) >= 30 else np.mean(diffs)
        returns = np.abs(np.diff(self.df["close"].values) / (self.df["close"].values[:-1] + 1e-10))
        entropy = -np.sum(returns * np.log(returns + 1e-10))

        # Config-driven weights
        weights = config.get("atr_entropy_weights", {})
        atr_weight, entropy_factor = weights.get(self.asset_class, (0.25, 0.05))
        threshold = max(0.0005, min(atr * atr_weight, np.mean(returns) + entropy * entropy_factor))

        filtered = [swings[0]]
        for i in range(1, len(swings)):
            if abs(swings[i][1] - filtered[-1][1]) / (filtered[-1][1] + 1e-10) >= threshold:
                filtered.append(swings[i])
        return filtered

class WaveValidator:
    """Validate Elliott Wave patterns."""

    def __init__(self, swings: List[Tuple[int, float]], organism: str) -> None:
        self.swings = swings
        self.organism = organism

    def validate_elliott_wave(self) -> bool:
        """Validate Elliott Wave structure."""
        try:
            if len(self.swings) < 6:
                return False
            wave1 = abs(self.swings[1][1] - self.swings[0][1])
            wave2 = abs(self.swings[2][1] - self.swings[1][1])
            wave3 = abs(self.swings[4][1] - self.swings[3][1])
            wave4 = abs(self.swings[5][1] - self.swings[4][1])

            expected_wave3 = wave1 * 1.618
            deviation3 = abs(expected_wave3 - wave3) / (expected_wave3 + 1e-6)
            retrace2 = wave2 / (wave1 + 1e-6)
            valid_wave3 = deviation3 < 0.15
            valid_wave2 = 0.382 <= retrace2 <= 0.618
            valid_wave4 = (wave4 / (wave3 + 1e-6)) < 1.0

            return valid_wave3 and valid_wave2 and valid_wave4
        except Exception as e:
            logger.warning(f"Elliott Wave validation failed: {e}")
            return False

class PhaseDeterminer:
    """Determine spiral phase based on swings and indicators."""

    def __init__(
        self,
        swings: List[Tuple[int, float]],
        main_wave: str,
        candle_patterns: Dict[str, Any],
        volatility: float,
        volume_change: float,
    ) -> None:
        self.swings = swings
        self.main_wave = main_wave
        self.candle_patterns = candle_patterns
        self.volatility = volatility
        self.volume_change = volume_change
        self.phase = "unknown"
        self.direction = "neutral"
        self.slope_strength = 0.0
        self.direction_confidence = 0.0

    def determine_phase(self) -> str:
        """Determine spiral phase based on indicators."""
        patterns = self.candle_patterns.get("patterns", [])
        start_price = self.swings[0][1]
        end_price = self.swings[-1][1]

        if any(p in patterns for p in ["hammer", "bullish_engulfing"]) and self.volatility < 0.3 and self.volume_change > 0:
            self.phase = "rebirth"
        elif any(p in patterns for p in ["bullish_engulfing", "three_white_soldiers"]) and end_price > start_price and self.volume_change > 0.1:
            self.phase = "growth"
        elif any(p in patterns for p in ["shooting_star", "hanging_man"]) and self.volatility > 0.5 and self.volume_change < 0:
            self.phase = "decay"
        elif any(p in patterns for p in ["bearish_engulfing", "three_black_crows"]) and end_price < start_price:
            self.phase = "death"
        else:
            self.phase = "unknown"
        return self.phase

    def determine_direction(self) -> str:
        """Determine market direction based on swing slopes."""
        indices = [i for i, _ in self.swings]
        prices = [p for _, p in self.swings]
        if len(prices) < 3:
            return "neutral"

        slope = np.polyfit(indices, prices, 1)[0]
        self.slope_strength = round(float(slope), 6)

        if slope > 0:
            self.direction = "bullish"
        elif slope < 0:
            self.direction = "bearish"
        else:
            self.direction = "neutral"

        abs_slope = abs(slope)
        thresholds = config.get("slope_confidence_thresholds", [0.003, 0.01, 0.02])
        self.direction_confidence = 1.0 if abs_slope > thresholds[2] else 0.7 if abs_slope > thresholds[1] else 0.4 if abs_slope > thresholds[0] else 0.1
        return self.direction

class SpiralPhaseScorer:
    """Compute spiral phase score."""

    def __init__(self, z_score: float, candle_confidence: float, sentiment_score: float) -> None:
        self.z_score = z_score
        self.candle_confidence = candle_confidence
        self.sentiment_score = sentiment_score

    def compute_score(self) -> float:
        """Compute weighted phase score."""
        sentiment_adj = (self.sentiment_score + 1) / 2
        return round(float(np.clip(
            0.6 * self.z_score + 0.25 * self.candle_confidence + 0.15 * sentiment_adj,
            0, 1
        )), 5)

class SpiralWaveEngineV2:
    """Rule-based engine for spiral phase annotation."""

    def __init__(self, organism: str, df: pd.DataFrame, sentiment_score: float = 0.0) -> None:
        if not isinstance(organism, str) or not organism:
            raise ValueError("Organism must be a non-empty string")
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("DataFrame must be non-empty")

        self.organism = organism.upper()
        self.df = df
        self.asset_class = self._determine_asset_class()
        self.sentiment_score = sentiment_score
        self.candle_lib = JapaneseCandleLibrary()

    def detect(self) -> Dict[str, Any]:
        """Detect spiral phase using rule-based logic."""
        try:
            swing_detector = SwingDetector(self.df, self.asset_class)
            swings = swing_detector.detect_swings()
            if len(swings) < 3:
                logger.warning(f"Insufficient swings for {self.organism}")
                return self._empty()

            main_wave = "impulse" if len(swings) % 2 == 0 else "corrective"
            sub_wave = f"wave{(len(swings) - 1) % 5 + 1}"
            micro_wave = f"wave{(len(swings) - 1) % 3 + 1}.micro"

            candle_features = self.candle_lib.detect(self.df.tail(60))
            volatility = self.df["close"].pct_change().std() * np.sqrt(config.get("annualization_factor", 252))
            volume_change = self.df["volume"].pct_change().mean()

            fibs = self._fibonacci(swings)
            z_score, validated = self._zscore(swings)

            phase_determiner = PhaseDeterminer(swings, main_wave, candle_features, volatility, volume_change)
            phase = phase_determiner.determine_phase()
            direction = phase_determiner.determine_direction()

            scorer = SpiralPhaseScorer(z_score, candle_features.get("confidence", 1.0), self.sentiment_score)
            spiral_phase_score = scorer.compute_score()

            return clean_for_json({
                "organism": self.organism,
                "phase": phase,
                "main_wave": main_wave,
                "sub_wave": sub_wave,
                "micro_wave": micro_wave,
                "z_score": z_score,
                "fib_levels": fibs,
                "validated": validated,
                "emerging_wave": not validated,
                "direction": direction,
                "slope_strength": phase_determiner.slope_strength,
                "direction_confidence": phase_determiner.direction_confidence,
                "spiral_phase_score": spiral_phase_score,
                "candle_patterns": candle_features.get("patterns", []),
                "volatility": float(volatility),
                "volume_change": float(volume_change)
            })
        except Exception as e:
            logger.error(f"SpiralWaveEngineV2 Error {self.organism}: {e}", exc_info=True)
            return self._empty()

    def _zscore(self, swings: List[Tuple[int, float]]) -> Tuple[float, bool]:
        """Compute z-score and validate Elliott Wave."""
        try:
            if len(swings) < 6:
                return 0.0, False
            wave1 = abs(swings[1][1] - swings[0][1])
            wave3 = abs(swings[4][1] - swings[3][1])
            expected = wave1 * 1.618
            z = 1 - abs(expected - wave3) / (expected + 1e-6)
            validator = WaveValidator(swings, self.organism)
            return round(float(z), 4), validator.validate_elliott_wave()
        except Exception as e:
            logger.warning(f"Z-score computation failed: {e}")
            return 0.0, False

    def _fibonacci(self, swings: List[Tuple[int, float]]) -> Dict[str, float]:
        """Compute Fibonacci levels."""
        if len(swings) < 3:
            return {}
        start, mid, end = swings[-3][1], swings[-2][1], swings[-1][1]
        rng = abs(mid - start)
        return {
            "0.382": round(mid - rng * 0.382, 6),
            "0.500": round(mid - rng * 0.5, 6),
            "0.618": round(mid - rng * 0.618, 6),
            "1.618": round(mid + rng * 1.618, 6),
        }

    def _determine_asset_class(self) -> str:
        """Determine asset class based on organism."""
        crypto_list = config.get("crypto_assets", ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD"])
        forex_suffixes = ["USD", "EUR", "JPY", "GBP", "AUD", "CHF", "NZD"]

        if self.organism in crypto_list:
            return "crypto"
        elif any(self.organism.endswith(suffix) for suffix in forex_suffixes):
            return "forex"
        else:
            return "stock"

    def _empty(self) -> Dict[str, Any]:
        """Return empty result dictionary."""
        return {
            "organism": self.organism,
            "phase": "unknown",
            "main_wave": "unknown",
            "sub_wave": "unknown",
            "micro_wave": "unknown",
            "z_score": 0.0,
            "fib_levels": {},
            "validated": False,
            "emerging_wave": False,
            "direction": "neutral",
            "slope_strength": 0.0,
            "direction_confidence": 0.0,
            "spiral_phase_score": 0.0,
            "candle_patterns": [],
            "volatility": 0.0,
            "volume_change": 0.0
        }
