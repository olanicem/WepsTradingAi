#!/usr/bin/env python3
# ==========================================================
# 🧬 WEPS SpiralPhaseDetectorStream — Integrated Wave Intelligence Edition
# ✅ Fuses real-time WEPS neurons with SpiralWaveEngine analysis
# ✅ Computes Bayesian confidence incorporating wave z-score & validation
# ✅ Correlates phase decisions with live market wave structures
# ✅ Logs neurons + waves to SpiralLifeMemory for immortal evolution
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import logging
import numpy as np
from time import sleep
from scipy.stats import norm
from weps.core.weps_pipeline import WEPSPipeline
from weps.memory.spiral_life_memory import SpiralLifeMemory
from weps.wave_engine.spiral_wave_engine import SpiralWaveEngine  # 🆕 Integrated wave engine

logger = logging.getLogger("WEPS.SpiralPhaseDetectorStream")

class SpiralPhaseDetectorStream:
    def __init__(self, organism: str, window_provider, refresh_rate: int = 60):
        """
        Args:
            organism (str): e.g., "EURUSD"
            window_provider (callable): returns latest rolling OHLCV DataFrame.
            refresh_rate (int): interval (seconds) between detection cycles.
        """
        self.organism = organism
        self.window_provider = window_provider
        self.refresh_rate = refresh_rate
        self.phase = "unknown"
        self.memory = SpiralLifeMemory(organism)
        self.latest_neurons = None
        self.latest_wave = None

        self.priors = {
            "impulse": norm(loc=0.4, scale=0.15),
            "wave_conf": norm(loc=0.5, scale=0.2),
            "volatility": norm(loc=0.5, scale=0.2),
            "entropy_slope": norm(loc=0.0, scale=0.001),
            "mutation": norm(loc=0.5, scale=0.2),
            "wave_z": norm(loc=0.5, scale=0.3),       # 🆕 wave z-score prior
        }

    def start(self):
        logger.info("[STREAM START] Spiral phase detection started for %s.", self.organism)
        try:
            while True:
                window_df = self.window_provider()
                if window_df is None or len(window_df) < 100:
                    logger.warning("[STREAM] Invalid or insufficient data. Skipping this iteration...")
                    sleep(self.refresh_rate)
                    continue

                detected_phase = self._detect_phase(window_df)
                logger.info("[STREAM] %s current phase: %s", self.organism, detected_phase)
                sleep(self.refresh_rate)
        except KeyboardInterrupt:
            logger.info("[STREAM STOP] Phase detection gracefully stopped for %s.", self.organism)

    def _detect_phase(self, window_df) -> str:
        try:
            # 🧠 Run WEPS neurons
            pipeline = WEPSPipeline(self.organism, window_df)
            pipeline.run()
            neurons = pipeline.neuron_outputs

            # Extract neuron metrics
            impulse = neurons.get("impulse", {}).get("impulse_score_norm", 0.0)
            wave_conf = neurons.get("elliott_wave", {}).get("wave_confidence", 0.0)
            volatility = neurons.get("volatility", {}).get("volatility_score_norm", 0.0)
            mutation = pipeline.gene_map.get("mutation", 0.0)
            entropy_slope_val = self._compute_entropy_slope(window_df)

            # 🌀 Run SpiralWaveEngine on same window
            wave_engine = SpiralWaveEngine(window_df)
            wave_result = wave_engine.detect()
            wave_z = wave_result.get("z_score", 0.0)
            wave_valid = wave_result.get("validated", False)

            # Compute Bayesian likelihoods
            likelihoods = {
                "impulse": self.priors["impulse"].pdf(impulse),
                "wave_conf": self.priors["wave_conf"].pdf(wave_conf),
                "volatility": self.priors["volatility"].pdf(volatility),
                "entropy_slope": self.priors["entropy_slope"].pdf(entropy_slope_val),
                "mutation": self.priors["mutation"].pdf(mutation),
                "wave_z": self.priors["wave_z"].pdf(wave_z),  # 🆕 include wave z-score
            }
            combined_confidence = np.mean(list(likelihoods.values()))

            logger.info("[REALTIME PHASE DETECT] %s: impulse=%.4f wave=%.4f vol=%.4f entropy=%.5f mutation=%.4f wave_z=%.4f | conf=%.4f",
                        self.organism, impulse, wave_conf, volatility, entropy_slope_val, mutation, wave_z, combined_confidence)

            # 🧬 Correlate neurons + waves in phase logic
            if combined_confidence > 0.6 and impulse > 0.6 and wave_z > 0.5 and wave_valid:
                self.phase = "growth"
            elif combined_confidence < 0.3 and volatility > 0.7 and not wave_valid:
                self.phase = "decay"
            elif entropy_slope_val > 0.001 and impulse < 0.2 and not wave_valid:
                self.phase = "death"
            elif impulse > 0.3 and entropy_slope_val < -0.001 and mutation > 0.5 and wave_z > 0.4:
                self.phase = "rebirth"
            elif 0.15 <= impulse <= 0.4 and combined_confidence > 0.4:
                self.phase = "maturity"
            else:
                self.phase = "maturity"

            # Update phase state + memory
            self.latest_neurons = neurons
            self.latest_wave = wave_result
            self.memory.append(self.phase, {"neurons": neurons, "wave": wave_result})
            return self.phase

        except Exception as e:
            logger.error("[ERROR] Phase detection failed in stream: %s", str(e))
            return "unknown"

    def _compute_entropy_slope(self, df) -> float:
        entropy_series = df["close"].pct_change().abs().rolling(10).mean().dropna()
        if len(entropy_series) < 10:
            return 0.0
        x = np.arange(10)
        y = entropy_series[-10:]
        slope, _ = np.polyfit(x, y, 1)
        return slope
