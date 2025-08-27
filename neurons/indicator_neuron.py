#!/usr/bin/env python3
# ================================================================
# 🧠 WEPS IndicatorNeuron — Full FESI Spiral Intelligence
# Author: Ola Bode | WEPS Creator
# Description:
#   - Multi-timeframe normalized indicator extraction (RSI, MACD, ATR, ADX, BB Width, Volume)
#   - Spiral phase-aware dynamic indicator interpretation using fuzzy weighting
#   - Mutation momentum & entropy slope metrics for market health detection
#   - Integration with Elliott Wave, Candle Pattern, Fibonacci outputs for conformance scoring
#   - Outputs robust indicator state with confidence, alerts, and phase alignment
# ================================================================

import numpy as np
import pandas as pd
import logging
import talib

logger = logging.getLogger("WEPS.Neurons.IndicatorNeuron")

class IndicatorNeuron:
    def __init__(self, 
                 dfs_multi_tf: dict,  # Dict[str, pd.DataFrame] with multi-timeframe OHLCV data
                 elliott_output: dict,
                 candle_output: dict,
                 fib_output: dict,
                 gene_map: dict,
                 spiral_phase: str = "neutral",
                 spiral_density: float = 0.0,
                 mutation_level: float = 0.0):
        """
        Initialize the IndicatorNeuron with all required data inputs.
        """
        self.dfs_multi_tf = dfs_multi_tf
        self.elliott_output = elliott_output or {}
        self.candle_output = candle_output or {}
        self.fib_output = fib_output or {}
        self.gene_map = gene_map or {}
        self.spiral_phase = spiral_phase.lower()
        self.spiral_density = spiral_density
        self.mutation_level = mutation_level

        # Define supported timeframes for indicator extraction
        self.supported_timeframes = ["1h", "4h", "1d"]
        self.indicators = ["rsi", "macd_hist", "atr", "adx", "bb_width", "volume"]

    def compute(self) -> dict:
        """
        Core compute method:
        - Extract indicators multi-timeframe
        - Compute entropy slope & mutation momentum
        - Contextualize indicators by spiral phase & mutation level
        - Compute spiral conformance score
        - Return rich indicator state dict
        """
        # 1️⃣ Extract multi-timeframe normalized indicators
        multi_tf_features = self._extract_multi_tf_indicators()

        # 2️⃣ Compute entropy slope & mutation momentum
        entropy_slope = self._compute_entropy_slope()
        mutation_momentum = self._compute_mutation_momentum()

        # 3️⃣ Adjust indicators dynamically based on spiral phase & mutation
        phase_adjusted_features = self._apply_phase_mutation_context(
            multi_tf_features, entropy_slope, mutation_momentum
        )

        # 4️⃣ Compute spiral conformance with internal wave, candle & fib outputs
        spiral_conformance = self._compute_spiral_conformance()

        # 5️⃣ Compose output dict
        indicator_state = {
            "features": phase_adjusted_features,          # Dict[str, float]
            "entropy_slope": entropy_slope,               # float [-1..1]
            "mutation_momentum": mutation_momentum,       # float [-1..1]
            "spiral_conformance": spiral_conformance,     # float [0..1]
            "phase": self.spiral_phase,                    # str
            "mutation_level": self.mutation_level,        # float [0..1]
        }

        logger.info(
            f"IndicatorNeuron completed: phase={self.spiral_phase} mutation_level={self.mutation_level:.3f} "
            f"spiral_conformance={spiral_conformance:.3f}"
        )
        return indicator_state

    def _extract_multi_tf_indicators(self) -> dict:
        """
        Extract and normalize indicators per timeframe.
        Returns a flattened dict with keys like '1h_rsi', '4h_macd_hist', etc.
        """

        features = {}

        for tf in self.supported_timeframes:
            df = self.dfs_multi_tf.get(tf)
            if df is None or df.empty:
                logger.warning(f"IndicatorNeuron: Missing or empty data for timeframe '{tf}'")
                continue

            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values

            # RSI 14
            rsi = talib.RSI(close, timeperiod=14)
            rsi_norm = self._normalize_last_value(rsi, 0, 100)
            features[f"{tf}_rsi"] = rsi_norm

            # MACD Histogram
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            macd_hist_norm = self._normalize_last_value(macd_hist, -0.05, 0.05)
            features[f"{tf}_macd_hist"] = macd_hist_norm

            # ATR 14
            atr = talib.ATR(high, low, close, timeperiod=14)
            atr_norm = self._normalize_last_value(atr, 0, np.max(atr) if np.max(atr) > 0 else 1)
            features[f"{tf}_atr"] = atr_norm

            # ADX 14
            adx = talib.ADX(high, low, close, timeperiod=14)
            adx_norm = self._normalize_last_value(adx, 0, 100)
            features[f"{tf}_adx"] = adx_norm

            # Bollinger Band Width
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            bb_width = (upper[-1] - lower[-1]) / middle[-1] if middle[-1] != 0 else 0.0
            bb_width_norm = np.clip(bb_width, 0, 1)
            features[f"{tf}_bb_width"] = bb_width_norm

            # Volume normalized by rolling max(20)
            vol_max = np.max(volume[-20:]) if len(volume) >= 20 else np.max(volume)
            vol_norm = self._normalize_last_value(volume, 0, vol_max)
            features[f"{tf}_volume"] = vol_norm

        return features

    def _compute_entropy_slope(self, window: int = 10) -> float:
        """
        Compute slope of entropy over last 'window' bars.
        Entropy is approximated here as standard deviation of returns.
        Returns slope normalized to [-1,1].
        """
        try:
            close = self.dfs_multi_tf.get("1d")["close"].values
            returns = np.diff(close) / close[:-1]
            entropy_series = pd.Series(returns).rolling(window=window).std()
            slope = self._linear_regression_slope(entropy_series.dropna().values)
            return np.clip(slope, -1.0, 1.0)
        except Exception as e:
            logger.warning(f"Failed entropy slope calc: {e}")
            return 0.0

    def _compute_mutation_momentum(self, window: int = 5) -> float:
        """
        Compute momentum of mutation level changes over 'window' periods.
        Returns slope normalized to [-1,1].
        """
        try:
            # Mutation time series assumed external; placeholder returns 0 for now
            return 0.0
        except Exception as e:
            logger.warning(f"Failed mutation momentum calc: {e}")
            return 0.0

    def _apply_phase_mutation_context(self, features: dict, entropy_slope: float, mutation_momentum: float) -> dict:
        """
        Adjust indicator values by spiral phase and mutation level using fuzzy logic:
        - Amplify signals in rebirth/growth
        - Damp signals in decay/death
        - Modulate uncertainty by mutation and entropy slope
        """
        adjusted = {}
        phase_weight_map = {
            "rebirth": 1.3,
            "growth": 1.2,
            "neutral": 1.0,
            "decay": 0.8,
            "death": 0.6
        }
        base_weight = phase_weight_map.get(self.spiral_phase, 1.0)
        mutation_factor = max(0.5, 1 - self.mutation_level)  # mutation dampens signal strength
        entropy_factor = 1 + entropy_slope  # entropy slope can amplify (>0) or dampen (<0)

        for k, v in features.items():
            val = np.clip(v * base_weight * mutation_factor * entropy_factor, 0.0, 1.0)
            adjusted[k] = round(val, 4)

        # Inject mutation and entropy momentum as separate features for AI consumption
        adjusted["entropy_slope"] = round(entropy_slope, 4)
        adjusted["mutation_momentum"] = round(mutation_momentum, 4)
        adjusted["mutation_level"] = round(self.mutation_level, 4)
        adjusted["spiral_phase_weight"] = round(base_weight, 4)

        return adjusted

    def _compute_spiral_conformance(self) -> float:
        """
        Compute a confidence score [0..1] indicating how well
        Elliott Wave, Candle Pattern, and Fibonacci outputs confirm the spiral phase.
        """
        try:
            elliott_phase = self.elliott_output.get("phase", "")
            candle_pattern = self.candle_output.get("detected_pattern", "")
            fib_levels = self.fib_output.get("fib_levels", {})

            score = 0.0
            weights = {"elliott": 0.4, "candle": 0.3, "fib": 0.3}

            if elliott_phase == self.spiral_phase:
                score += weights["elliott"]

            bullish_patterns = ["bullish_engulfing", "morning_star", "piercing_line", "hammer", "tweezer_bottom"]
            bearish_patterns = ["bearish_engulfing", "evening_star", "dark_cloud_cover", "shooting_star", "tweezer_top"]

            if self.spiral_phase in ["rebirth", "growth"]:
                if candle_pattern in bullish_patterns:
                    score += weights["candle"]
            elif self.spiral_phase in ["decay", "death"]:
                if candle_pattern in bearish_patterns:
                    score += weights["candle"]
            else:
                score += weights["candle"] * 0.5  # neutral partial credit

            if fib_levels:
                key_levels = [0.382, 0.5, 0.618]
                proximity_scores = []
                for level in key_levels:
                    if level in fib_levels:
                        val = fib_levels[level]
                        close = self.dfs_multi_tf.get("1d")["close"].iloc[-1]
                        proximity = max(0, 1 - abs(close - val) / close)
                        proximity_scores.append(proximity)
                if proximity_scores:
                    score += weights["fib"] * np.mean(proximity_scores)

            return min(score, 1.0)
        except Exception as e:
            logger.warning(f"Failed spiral conformance calc: {e}")
            return 0.0

    def _normalize_last_value(self, arr, min_val, max_val):
        try:
            val = None
            for v in reversed(arr):
                if v is not None and not np.isnan(v):
                    val = v
                    break
            if val is None:
                return 0.0
            norm = (val - min_val) / (max_val - min_val + 1e-8)
            return min(max(norm, 0.0), 1.0)
        except Exception as e:
            logger.warning(f"Normalization failed: {e}")
            return 0.0

    def _linear_regression_slope(self, y: np.ndarray) -> float:
        try:
            x = np.arange(len(y))
            if len(y) < 2:
                return 0.0
            A = np.vstack([x, np.ones(len(x))]).T
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            return slope
        except Exception as e:
            logger.warning(f"Linear regression slope calc failed: {e}")
            return 0.0
