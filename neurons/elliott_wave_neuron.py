#!/usr/bin/env python3
# ==========================================================
# 🌊 WEPS ElliottWaveNeuron v7.0 — Robust Swing & Wave Detection
# Author: Ola Bode (WEPS Creator)
# Description:
#   - Uses ATR for volatility filtering
#   - Uses scipy peak detection with prominence for swing extraction
#   - Cleans candidate wave sequences (no consecutive peaks/valleys)
#   - Validates impulse and corrective waves with tolerance
#   - Outputs full dict compatible with ImpulseNeuron interface
# ==========================================================
import logging
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

logger = logging.getLogger("WEPS.Neurons.ElliottWaveNeuron")

class ElliottWaveNeuron:
    def __init__(self, df: pd.DataFrame, phase="neutral", config=None):
        self.df = df
        self.phase = phase
        self.config = config or {}

    def compute(self) -> dict:
        try:
            candidate_waves = self._extract_candidate_waves(self.df)
            candidate_waves = self._clean_consecutive_waves(candidate_waves)
            return self.confirm(candidate_waves)
        except Exception as e:
            logger.error("[ElliottWaveNeuron] compute() failed: %s", str(e), exc_info=True)
            return self._empty_result()

    def _empty_result(self) -> dict:
        return {
            "impulse_waves": [],
            "corrective_waves": [],
            "wave_probability_score": 0.0,
            "wave_confidence": 0.0,
            "valid_impulse": False,
            "total_candidate_waves": 0,
            "phase": self.phase,
        }

    def _extract_candidate_waves(self, df: pd.DataFrame) -> list:
        if df is None or df.empty or len(df) < 10:
            logger.warning("[ElliottWaveNeuron] DataFrame too small for wave extraction")
            return []

        # Calculate ATR for volatility filtering
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1)
        atr = tr.max(axis=1).rolling(window=14, min_periods=1).mean()

        price = (df['high'] + df['low']) / 2

        # Peak detection with prominence scaled by ATR
        prominence = atr.fillna(atr.mean()).values * 0.5

        # Detect peaks (local maxima)
        peak_indices, _ = find_peaks(price.values, prominence=prominence)
        # Detect valleys (local minima) by inverting price
        valley_indices, _ = find_peaks(-price.values, prominence=prominence)

        candidate_waves = []
        for idx in peak_indices:
            candidate_waves.append({"index": idx, "type": "peak", "price": float(df['high'].iloc[idx])})
        for idx in valley_indices:
            candidate_waves.append({"index": idx, "type": "valley", "price": float(df['low'].iloc[idx])})

        # Sort by index ascending
        candidate_waves.sort(key=lambda x: x['index'])

        logger.debug(f"[ElliottWaveNeuron] Extracted {len(candidate_waves)} candidate waves after ATR & prominence filtering")
        return candidate_waves

    def _clean_consecutive_waves(self, waves: list) -> list:
        """Remove consecutive waves of the same type by keeping the stronger/extreme one"""
        if not waves:
            return waves
        cleaned = [waves[0]]
        for w in waves[1:]:
            if w['type'] == cleaned[-1]['type']:
                # For peaks keep higher price; for valleys keep lower price
                if w['type'] == 'peak':
                    if w['price'] > cleaned[-1]['price']:
                        cleaned[-1] = w
                else:  # valley
                    if w['price'] < cleaned[-1]['price']:
                        cleaned[-1] = w
            else:
                cleaned.append(w)
        logger.debug(f"[ElliottWaveNeuron] Cleaned waves: {len(cleaned)} after removing consecutive duplicates")
        return cleaned

    def confirm(self, candidate_waves: list) -> dict:
        if not isinstance(candidate_waves, list) or not all(isinstance(w, dict) for w in candidate_waves):
            logger.error("[ElliottWaveNeuron] Invalid candidate_waves input: %s", candidate_waves)
            return self._empty_result()

        impulse = []
        corrective = []
        valid_impulse = False
        valid_corrective = False

        # Relaxed impulse validation with sequence tolerance
        for i in range(len(candidate_waves) - 4):
            seq = [w['type'] for w in candidate_waves[i:i+5]]
            if self._is_valid_impulse_sequence(seq):
                impulse = candidate_waves[i:i+5]
                valid_impulse = True
                break

        # Relaxed corrective validation
        for i in range(len(candidate_waves) - 2):
            seq = [w['type'] for w in candidate_waves[i:i+3]]
            if self._is_valid_corrective_sequence(seq):
                corrective = candidate_waves[i:i+3]
                valid_corrective = True
                break

        impulse = self._enrich_wave_times(impulse)
        corrective = self._enrich_wave_times(corrective)

        wave_confidence = self._compute_wave_confidence(impulse) if valid_impulse else 0.0
        wave_probability_score = self._compute_wave_probability(impulse) if valid_impulse else 0.0
        adjusted_confidence = round(self._adjust_score_by_phase(wave_confidence), 4)

        result = {
            "impulse_waves": impulse,
            "corrective_waves": corrective,
            "wave_probability_score": round(wave_probability_score, 4),
            "wave_confidence": adjusted_confidence,
            "valid_impulse": valid_impulse,
            "total_candidate_waves": len(candidate_waves),
            "phase": self.phase,
        }
        logger.info("[ElliottWaveNeuron] Confirmation result: %s", result)
        return result

    def _is_valid_impulse_sequence(self, seq: list) -> bool:
        # Allow small tolerance: standard Elliott impulse sequences or close variations
        valid_patterns = [
            ['peak', 'valley', 'peak', 'valley', 'peak'],
            ['valley', 'peak', 'valley', 'peak', 'valley'],
        ]
        if seq in valid_patterns:
            return True
        # Allow 1 mismatch tolerance
        mismatches = [s != p for s, p in zip(seq, valid_patterns[0])]
        if sum(mismatches) <= 1:
            return True
        mismatches = [s != p for s, p in zip(seq, valid_patterns[1])]
        if sum(mismatches) <= 1:
            return True
        return False

    def _is_valid_corrective_sequence(self, seq: list) -> bool:
        valid_patterns = [
            ['peak', 'valley', 'peak'],
            ['valley', 'peak', 'valley'],
        ]
        if seq in valid_patterns:
            return True
        # Allow 1 mismatch tolerance
        mismatches = [s != p for s, p in zip(seq, valid_patterns[0])]
        if sum(mismatches) <= 1:
            return True
        mismatches = [s != p for s, p in zip(seq, valid_patterns[1])]
        if sum(mismatches) <= 1:
            return True
        return False

    def _compute_wave_confidence(self, impulse: list) -> float:
        if len(impulse) < 5:
            return 0.0
        prices = [w['price'] for w in impulse]
        if any(np.isnan(p) for p in prices):
            return 0.0
        wave1, wave3, wave5 = impulse[0], impulse[2], impulse[4]
        dist_13 = abs(wave3['price'] - wave1['price'])
        dist_35 = abs(wave5['price'] - wave3['price'])
        expected_ratio = 1.618
        measured_ratio = dist_35 / (dist_13 + 1e-9)
        z = (measured_ratio - expected_ratio) / 0.2
        return np.clip(1 - abs(z), 0, 1)

    def _compute_wave_probability(self, impulse: list) -> float:
        if len(impulse) < 5:
            return 0.0
        prices = np.array([w['price'] for w in impulse], dtype=float)
        diffs = np.abs(np.diff(prices))
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        if std_diff == 0:
            return 1.0
        prob = max(0.0, 1 - (std_diff / (mean_diff + 1e-9)))
        return np.clip(prob, 0, 1)

    def _enrich_wave_times(self, waves: list) -> list:
        if not waves:
            return waves
        enriched = []
        for w in waves:
            idx = w.get('index')
            if idx is None or self.df is None or self.df.empty:
                w['time'] = float(idx) if idx is not None else 0.0
            else:
                try:
                    ts = self.df.index[idx]
                    w['time'] = float(ts.timestamp()) if hasattr(ts, 'timestamp') else float(ts)
                except Exception as e:
                    logger.warning("[ElliottWaveNeuron] Failed to enrich wave time at index %s: %s", idx, str(e))
                    w['time'] = float(idx)
            enriched.append(w)
        return enriched

    def _adjust_score_by_phase(self, score: float) -> float:
        phase_weights = {"rebirth": 1.2, "growth": 1.0, "decay": 0.8, "neutral": 1.0}
        return np.clip(score * phase_weights.get(self.phase, 1.0), 0, 1)
