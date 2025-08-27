#!/usr/bin/env python3
# ==========================================================
# 🌊 WEPS SpiralWaveEngine v2.1 — Directional Intelligence Upgrade
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Adds slope vector detection and trend confidence scoring
#   - Enhances phase/direction determination using wave geometry
#   - Modular for Reflex Cortex + Decision Fuser
# ==========================================================

import logging
import numpy as np
import pandas as pd
import json
from scipy.signal import find_peaks
from typing import List, Tuple, Dict, Any

logger = logging.getLogger("WEPS.SpiralWaveEngineV2")

def clean_for_json(obj: Any) -> Any:
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
    def __init__(self, df: pd.DataFrame, asset_class: str):
        self.df = df
        self.asset_class = asset_class

    def detect_swings(self) -> List[Tuple[int, float]]:
        closes = self.df["close"].values
        if len(closes) < 30:
            return []

        peaks, _ = find_peaks(closes, distance=5)
        troughs, _ = find_peaks(-closes, distance=5)
        indices = np.sort(np.concatenate((peaks, troughs)))
        swings = [(i, closes[i]) for i in indices if i < len(closes)]

        return self._filter_swings(swings)

    def _filter_swings(self, swings: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        if len(swings) < 2:
            return swings

        closes = [p for _, p in swings]
        diffs = np.abs(np.diff(closes))
        atr = np.mean(diffs[-30:]) if len(diffs) >= 30 else np.mean(diffs)
        returns = np.abs(np.diff(self.df["close"].values) / (self.df["close"].values[:-1] + 1e-10))
        entropy = -np.sum(returns * np.log(returns + 1e-10))

        atr_weight, entropy_factor = {"forex": (0.25, 0.05), "crypto": (0.4, 0.1), "stock": (0.15, 0.03)}.get(
            self.asset_class, (0.25, 0.05)
        )
        threshold = max(0.0005, min(atr * atr_weight, np.mean(returns) + entropy * entropy_factor))

        filtered = [swings[0]]
        for i in range(1, len(swings)):
            if abs(swings[i][1] - filtered[-1][1]) / (filtered[-1][1] + 1e-10) >= threshold:
                filtered.append(swings[i])
        return filtered

class WaveValidator:
    def __init__(self, swings: List[Tuple[int, float]], organism: str):
        self.swings = swings
        self.organism = organism

    def validate_elliott_wave(self) -> bool:
        try:
            wave1 = abs(self.swings[1][1] - self.swings[0][1])
            wave2 = abs(self.swings[2][1] - self.swings[1][1])
            wave3 = abs(self.swings[4][1] - self.swings[3][1])
            wave4 = abs(self.swings[5][1] - self.swings[4][1])
        except IndexError:
            return False

        expected_wave3 = wave1 * 1.618
        deviation3 = abs(expected_wave3 - wave3) / (expected_wave3 + 1e-6)
        retrace2 = wave2 / (wave1 + 1e-6)
        valid_wave3 = deviation3 < 0.15
        valid_wave2 = 0.382 <= retrace2 <= 0.618
        valid_wave4 = (wave4 / (wave3 + 1e-6)) < 1.0

        return valid_wave3 and valid_wave2 and valid_wave4

class PhaseDeterminer:
    def __init__(self, swings: List[Tuple[int, float]], main_wave: str):
        self.swings = swings
        self.main_wave = main_wave
        self.phase = "unknown"
        self.direction = "neutral"
        self.slope_strength = 0.0
        self.direction_confidence = 0.0

    def determine_phase(self) -> str:
        start_price = self.swings[0][1]
        end_price = self.swings[-1][1]
        if end_price > start_price:
            self.phase = "growth" if self.main_wave == "impulse" else "rebirth"
        elif end_price < start_price:
            self.phase = "decay"
        else:
            self.phase = "maturity"
        return self.phase

    def determine_direction(self) -> str:
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
        self.direction_confidence = (
            1.0 if abs_slope > 0.02 else 0.7 if abs_slope > 0.01 else 0.4 if abs_slope > 0.003 else 0.1
        )
        return self.direction

class SpiralPhaseScorer:
    def __init__(self, z_score: float, candle_confidence: float, sentiment_score: float):
        self.z_score = z_score
        self.candle_confidence = candle_confidence
        self.sentiment_score = sentiment_score

    def compute_score(self) -> float:
        sentiment_adj = (self.sentiment_score + 1) / 2
        return round(float(np.clip(
            0.6 * self.z_score + 0.25 * self.candle_confidence + 0.15 * sentiment_adj,
            0, 1
        )), 5)

class SpiralWaveEngineV2:
    def __init__(self, organism: str, df: pd.DataFrame, candle_pattern_confidence: float = 0.0, sentiment_score: float = 0.0):
        self.organism = organism.upper()
        self.df = df
        self.asset_class = self._determine_asset_class()
        self.candle_pattern_confidence = candle_pattern_confidence
        self.sentiment_score = sentiment_score

    def detect(self) -> Dict[str, Any]:
        try:
            swing_detector = SwingDetector(self.df, self.asset_class)
            swings = swing_detector.detect_swings()
            if len(swings) < 3:
                return self._empty()

            main_wave = "impulse" if len(swings) % 2 == 0 else "corrective"
            sub_wave = f"wave{(len(swings) - 1) % 5 + 1}"
            micro_wave = f"wave{(len(swings) - 1) % 3 + 1}.micro"

            fibs = self._fibonacci(swings)
            z_score, validated = self._zscore(swings)

            phase_determiner = PhaseDeterminer(swings, main_wave)
            phase = phase_determiner.determine_phase()
            direction = phase_determiner.determine_direction()

            scorer = SpiralPhaseScorer(z_score, self.candle_pattern_confidence, self.sentiment_score)
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
                "spiral_phase_score": spiral_phase_score
            })
        except Exception as e:
            logger.error(f"[WEPS.WaveEngine] Error: {str(e)}", exc_info=True)
            return self._empty()

    def _zscore(self, swings) -> Tuple[float, bool]:
        try:
            if len(swings) < 6:
                return 0.0, False
            wave1 = abs(swings[1][1] - swings[0][1])
            wave3 = abs(swings[4][1] - swings[3][1])
            expected = wave1 * 1.618
            z = 1 - abs(expected - wave3) / (expected + 1e-6)
            validator = WaveValidator(swings, self.organism)
            return round(float(z), 4), validator.validate_elliott_wave()
        except:
            return 0.0, False

    def _fibonacci(self, swings) -> Dict[str, float]:
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
        if self.organism.endswith(("USD", "EUR", "JPY", "GBP", "AUD", "CHF", "NZD")):
            return "forex"
        elif self.organism.startswith("BTC") or self.organism.endswith("USD") and len(self.organism) == 6:
            return "crypto"
        else:
            return "stock"

    def _empty(self) -> Dict[str, Any]:
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
        }
