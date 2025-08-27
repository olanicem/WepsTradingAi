#!/usr/bin/env python3
# ============================================================================
# 🕯️ WEPS CandlePatternNeuron v10.2 — Apex Candlestick Reflex Intelligence
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Detects 15+ Japanese candlestick patterns (reversal & continuation)
#   - Applies trap detection logic (volume/impulse collapse)
#   - Spiral-phase scoring and financial signal validation
# ============================================================================
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("WEPS.Neurons.CandlePatternNeuron")

class CandlePatternNeuron:
    def __init__(self, phase: str = "unknown"):
        self.phase = phase
        self.df: Optional[pd.DataFrame] = None
        logger.info(f"[CandlePatternNeuron] Initialized with spiral phase: {phase}")

    def compute(self, df: pd.DataFrame = None, current_phase: str = None) -> Dict[str, Any]:
        if df is not None:
            self.df = df
        elif self.df is None:
            raise ValueError("No candle DataFrame provided or cached.")

        df = self.df
        phase = current_phase or self.phase

        if len(df) < 5:
            logger.warning("[CandlePatternNeuron] Insufficient candles.")
            return self._fallback()

        p5, p4, p3, p2, p1 = df.iloc[-5:]

        pattern = "none"
        classification = "neutral"
        trap = False
        score = 0.0

        # === Pattern Matching (Reversal + Continuation)
        pattern_funcs = [
            ("bullish_engulfing", self._is_bullish_engulfing),
            ("bearish_engulfing", self._is_bearish_engulfing),
            ("hammer", self._is_hammer),
            ("shooting_star", self._is_shooting_star),
            ("morning_star", self._is_morning_star),
            ("evening_star", self._is_evening_star),
            ("tweezer_bottom", self._is_tweezer_bottom),
            ("tweezer_top", self._is_tweezer_top),
            ("doji", self._is_doji),
            ("three_white_soldiers", lambda: self._is_three_white_soldiers(p3, p2, p1)),
            ("three_black_crows", lambda: self._is_three_black_crows(p3, p2, p1)),
            ("rising_three", lambda: self._is_rising_three(p5, p4, p3, p2, p1)),
            ("falling_three", lambda: self._is_falling_three(p5, p4, p3, p2, p1)),
            ("upside_tasuki_gap", lambda: self._is_upside_tasuki(p3, p2, p1)),
            ("downside_tasuki_gap", lambda: self._is_downside_tasuki(p3, p2, p1)),
        ]

        for name, fn in pattern_funcs:
            try:
                if fn():
                    pattern = name
                    classification = self._classify_pattern(name)
                    score = self._score_pattern(p1, p2, df['volume'])
                    trap = self._detect_trap(p1, df['volume'])
                    break
            except Exception as e:
                logger.debug(f"[CandlePatternNeuron] Error in pattern {name}: {e}")

        logger.info(f"🕯️ Pattern: {pattern} | Score: {score:.2f} | Class: {classification} | Trap: {trap} | Phase: {phase}")

        return {
            "pattern_name": pattern,
            "pattern_class": classification,
            "pattern_score": round(score, 4),
            "spiral_phase": phase,
            "trap_signal": trap
        }

    # === Pattern Functions ===
    def _is_bullish_engulfing(self):
        p1, p2 = self.df.iloc[-2], self.df.iloc[-1]
        return p1['close'] < p1['open'] and p2['close'] > p2['open'] and p2['close'] > p1['open'] and p2['open'] < p1['close']

    def _is_bearish_engulfing(self):
        p1, p2 = self.df.iloc[-2], self.df.iloc[-1]
        return p1['close'] > p1['open'] and p2['close'] < p2['open'] and p2['open'] > p1['close'] and p2['close'] < p1['open']

    def _is_hammer(self):
        c = self.df.iloc[-1]
        body = abs(c['close'] - c['open'])
        return (c['high'] - c['low']) > 2 * body and (c['close'] - c['low']) / (c['high'] - c['low'] + 1e-6) > 0.6

    def _is_shooting_star(self):
        c = self.df.iloc[-1]
        body = abs(c['close'] - c['open'])
        return (c['high'] - c['low']) > 2 * body and (c['high'] - c['close']) / (c['high'] - c['low'] + 1e-6) > 0.6

    def _is_morning_star(self):
        p3, p2, p1 = self.df.iloc[-3:]
        return p3['close'] < p3['open'] and p2['close'] < p2['open'] and p1['close'] > p1['open'] and p1['close'] > (p2['open'] + p2['close']) / 2

    def _is_evening_star(self):
        p3, p2, p1 = self.df.iloc[-3:]
        return p3['close'] > p3['open'] and p2['close'] > p2['open'] and p1['close'] < p1['open'] and p1['close'] < (p2['open'] + p2['close']) / 2

    def _is_tweezer_bottom(self):
        p2, p1 = self.df.iloc[-2:]
        return abs(p2['low'] - p1['low']) < 0.001 * p2['low']

    def _is_tweezer_top(self):
        p2, p1 = self.df.iloc[-2:]
        return abs(p2['high'] - p1['high']) < 0.001 * p2['high']

    def _is_doji(self):
        c = self.df.iloc[-1]
        return abs(c['close'] - c['open']) <= 0.1 * (c['high'] - c['low'])

    def _is_three_white_soldiers(self, p3, p2, p1):
        return all(c['close'] > c['open'] for c in [p3, p2, p1])

    def _is_three_black_crows(self, p3, p2, p1):
        return all(c['close'] < c['open'] for c in [p3, p2, p1])

    def _is_rising_three(self, p5, p4, p3, p2, p1):
        return p5['close'] > p5['open'] and all(c['close'] < c['open'] for c in [p4, p3, p2]) and p1['close'] > p1['open'] and p1['close'] > p5['close']

    def _is_falling_three(self, p5, p4, p3, p2, p1):
        return p5['close'] < p5['open'] and all(c['close'] > c['open'] for c in [p4, p3, p2]) and p1['close'] < p1['open'] and p1['close'] < p5['close']

    def _is_upside_tasuki(self, p3, p2, p1):
        return p3['close'] > p3['open'] and p2['open'] > p3['close'] and p2['close'] > p2['open'] and p1['open'] > p2['close'] and p1['close'] < p1['open']

    def _is_downside_tasuki(self, p3, p2, p1):
        return p3['close'] < p3['open'] and p2['open'] < p3['close'] and p2['close'] < p2['open'] and p1['open'] < p2['close'] and p1['close'] > p1['open']

    # === Classification Logic ===
    def _classify_pattern(self, name: str) -> str:
        if name in {"bullish_engulfing", "morning_star", "tweezer_bottom", "hammer", "three_white_soldiers"}:
            return "bullish_reversal"
        elif name in {"bearish_engulfing", "evening_star", "tweezer_top", "shooting_star", "three_black_crows"}:
            return "bearish_reversal"
        elif name in {"rising_three", "upside_tasuki_gap"}:
            return "bullish_continuation"
        elif name in {"falling_three", "downside_tasuki_gap"}:
            return "bearish_continuation"
        elif name == "doji":
            return "indecision"
        else:
            return "neutral"

    def _score_pattern(self, p1: pd.Series, p2: pd.Series, volume: pd.Series) -> float:
        body_ratio = abs(p2['close'] - p2['open']) / (abs(p1['close'] - p1['open']) + 1e-6)
        volume_ratio = p2['volume'] / (volume.iloc[-5:].mean() + 1e-6)
        phase_boost = {"rebirth": 1.3, "growth": 1.1, "decay": 0.7, "death": 0.4}
        boost = phase_boost.get(self.phase, 1.0)
        return min(1.0, (0.6 * body_ratio + 0.4 * volume_ratio) * boost)

    def _detect_trap(self, p1: pd.Series, volume: pd.Series) -> bool:
        avg_vol = volume.iloc[-5:].mean()
        return p1['volume'] < 0.5 * avg_vol

    def _fallback(self):
        return {
            "pattern_name": "none",
            "pattern_class": "neutral",
            "pattern_score": 0.0,
            "spiral_phase": self.phase,
            "trap_signal": False
        }

