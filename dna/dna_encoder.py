#!/usr/bin/env python3
# ===============================================================
# 🧬 WEPS DNAEncoder v4.0 — Institutional Spiral Intelligence Edition with Fractal & Extra Features
# File: weps/dna/dna_encoder.py
# Author: Ola | WEPS Creator
# Description:
#   - Encodes OHLCV data into a normalized 40D DNA vector
#   - Adds biological fractal features (Hurst exponent, fractal dimension, wavelet energies)
#   - Accepts and incorporates extra numerical features (e.g. multi-TF correlations)
#   - Spiral-aware biological intelligence encoding
#   - Institutional-grade engineering and logging
# ===============================================================

import numpy as np
import pandas as pd
import pywt  # PyWavelets for wavelet transforms
import logging

logger = logging.getLogger("WEPS.DNAEncoder")

class DNAEncoder:
    def __init__(self, df: pd.DataFrame, config: dict, extra_features: dict = None):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("DNAEncoder requires a pandas DataFrame.")
        if len(df) < 7:
            raise ValueError("DNAEncoder requires at least 7 rows for full DNA encoding.")
        self.df = df
        self.config = config
        self.extra_features = extra_features or {}

    def encode(self) -> np.ndarray:
        logger.info("DNAEncoder v4.0 encoding started with fractal and extra features...")

        vector = []

        # Extract features for windows 1, 3, 7 days
        windows = [1, 3, 7]
        for w in windows:
            window_df = self.df.iloc[-w:]
            vector.extend(self._extract_features(window_df))

        # Fractal features on close prices last 50 candles or max available
        close_series = self.df['close'].iloc[-50:] if len(self.df) >= 50 else self.df['close']

        # 1) Hurst exponent
        hurst = self._compute_hurst_exponent(close_series)
        vector.append(self._normalize(hurst, feature="hurst"))

        # 2) Petrosian fractal dimension
        fractal_dim = self._compute_petrosian_fd(close_series)
        vector.append(self._normalize(fractal_dim, feature="fractal_dim"))

        # 3) Wavelet energy coefficients (3 detail levels)
        wavelet_energies = self._compute_wavelet_energies(close_series)
        vector.extend([self._normalize(e, feature="wavelet_energy") for e in wavelet_energies])

        # Meta-spiral signals
        entropy = float(np.std(vector))
        sei = float(np.mean(vector))
        impulse_total = max(vector) - min(vector)
        phase_score = self._compute_spiral_phase(sei, entropy)

        vector.extend([
            self._normalize(entropy, "entropy"),
            self._normalize(sei, "sei"),
            self._normalize(impulse_total, "impulse"),
            phase_score
        ])

        # Spiral lifecycle metrics
        vector.extend([
            self._compute_trend(),
            self._compute_spiral_loop(),
            self._compute_mutation_delta(),
            self._compute_entropy_delta(),
            self._compute_reflex_flag(),
            self._compute_half_life_position()
        ])

        # Append normalized extra features (sorted keys for consistent order)
        if self.extra_features:
            for key in sorted(self.extra_features.keys()):
                val = self.extra_features[key]
                norm_val = self._normalize(val, "impulse")  # Or customize normalization per feature if needed
                vector.append(norm_val)

        # Clamp vector to exactly 40 features
        if len(vector) < 40:
            vector.extend([0.0] * (40 - len(vector)))
        elif len(vector) > 40:
            vector = vector[:40]

        dna_vector = np.array(vector, dtype=np.float32)
        dna_vector = np.nan_to_num(dna_vector, nan=0.0, posinf=1.0, neginf=0.0)
        dna_vector = np.clip(dna_vector, 0.0, 1.0)

        logger.info(f"DNAEncoder v4.0 completed. DNA vector shape: {dna_vector.shape}")
        return dna_vector

    def _extract_features(self, window: pd.DataFrame) -> list:
        o = window['open'].mean()
        h = window['high'].max()
        l = window['low'].min()
        c = window['close'].iloc[-1]
        v = window['volume'].mean()

        return [
            self._normalize(o, "price"),
            self._normalize(h, "price"),
            self._normalize(l, "price"),
            self._normalize(c, "price"),
            self._normalize(v, "volume"),
            self._normalize((c - o) / (o + 1e-8), "momentum"),
            self._normalize(h - l, "volatility"),
            self._normalize((c - l) / (h - l + 1e-8), "impulse"),
            self._normalize((h + l + c) / 3, "price"),
            self._normalize((c - o) / (h - l + 1e-8), "metabolic"),
        ]

    def _compute_hurst_exponent(self, series: pd.Series) -> float:
        ts = series.values
        N = len(ts)
        if N < 20:
            return 0.5  # Random walk baseline
        lags = range(2, min(20, N // 2))
        tau = []
        for lag in lags:
            diff_series = ts[lag:] - ts[:-lag]
            tau.append(np.sqrt(np.std(diff_series)))
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst_exp = poly[0] * 2.0
        hurst_clamped = max(0.0, min(hurst_exp, 1.0))
        logger.debug(f"Computed Hurst exponent: {hurst_clamped}")
        return hurst_clamped

    def _compute_petrosian_fd(self, series: pd.Series) -> float:
        ts = series.values
        N = len(ts)
        diff_sign = np.diff(np.sign(np.diff(ts)))
        N_delta = np.sum(diff_sign != 0)
        if N_delta == 0 or N <= 1:
            return 1.0
        petrosian = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * N_delta)))
        petrosian_clamped = max(1.0, min(petrosian, 2.0))  # FD ~ 1–2
        fractal_norm = (petrosian_clamped - 1.0) / 1.0
        logger.debug(f"Computed Petrosian fractal dimension: {fractal_norm}")
        return fractal_norm

    def _compute_wavelet_energies(self, series: pd.Series) -> list:
        try:
            coeffs = pywt.wavedec(series.values, 'db4', level=3)
            energies = [np.sum(np.square(c)) / len(c) for c in coeffs[1:]]
            max_energy = max(energies) if energies else 1.0
            normalized_energies = [e / max_energy for e in energies]
            logger.debug(f"Computed wavelet energies: {normalized_energies}")
            return normalized_energies
        except Exception as e:
            logger.warning(f"Wavelet energy computation failed: {str(e)}")
            return [0.0, 0.0, 0.0]

    def _normalize(self, value: float, feature: str) -> float:
        norm_ranges = {
            "hurst": (0.0, 1.0),
            "fractal_dim": (0.0, 1.0),
            "wavelet_energy": (0.0, 1.0),
            "price": (0.8, 1.6),
            "volume": (1e5, 5e6),
            "momentum": (-0.1, 0.1),
            "volatility": (0.001, 0.05),
            "impulse": (0.0, 1.0),
            "metabolic": (-0.1, 0.1),
            "entropy": (0.01, 0.5),
            "sei": (0.1, 1.0),
            "mutation_delta": (-0.05, 0.05),
            "entropy_delta": (-0.05, 0.05),
        }
        min_val, max_val = norm_ranges.get(feature, (0.0, 1.0))
        normalized = (value - min_val) / (max_val - min_val + 1e-8)
        normalized_clamped = max(0.0, min(normalized, 1.0))
        logger.debug(f"Normalized feature {feature}: raw={value} norm={normalized_clamped}")
        return normalized_clamped

    def _compute_spiral_phase(self, sei, entropy) -> float:
        if sei > 0.75 and entropy < 0.3:
            return 0.2  # Growth
        elif sei > 0.5 and entropy < 0.5:
            return 0.4  # Maturity
        elif sei > 0.3:
            return 0.6  # Decay
        elif sei > 0.1:
            return 0.8  # Death
        return 0.0  # Rebirth

    def _compute_trend(self) -> int:
        if self.df['close'].iloc[-1] > self.df['close'].iloc[-3]:
            return 1
        elif self.df['close'].iloc[-1] < self.df['close'].iloc[-3]:
            return -1
        return 0

    def _compute_spiral_loop(self) -> int:
        return len(self.df) % 5

    def _compute_mutation_delta(self) -> float:
        mom_now = (self.df['close'].iloc[-1] - self.df['open'].iloc[-1]) / (self.df['open'].iloc[-1] + 1e-8)
        mom_past = (self.df['close'].iloc[-7] - self.df['open'].iloc[-7]) / (self.df['open'].iloc[-7] + 1e-8)
        return self._normalize(mom_now - mom_past, "mutation_delta")

    def _compute_entropy_delta(self, window_3=None, window_7=None) -> float:
        if window_3 is None:
            window_3 = self.df.iloc[-3:]
        if window_7 is None:
            window_7 = self.df.iloc[-7:]
        vol_3d = window_3['high'].max() - window_3['low'].min()
        vol_7d = window_7['high'].max() - window_7['low'].min()
        return self._normalize(vol_7d - vol_3d, "entropy_delta")

    def _compute_reflex_flag(self) -> int:
        impulse_last = (self.df['close'].iloc[-1] - self.df['low'].iloc[-1]) / (self.df['high'].iloc[-1] - self.df['low'].iloc[-1] + 1e-8)
        return 1 if impulse_last > 0.8 else 0

    def _compute_half_life_position(self) -> float:
        lifespan = self.config.get("lifespan_days", 20)
        position = len(self.df) % lifespan
        return self._normalize(position, "entropy_delta")
