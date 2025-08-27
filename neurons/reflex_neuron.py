#!/usr/bin/env python3
# ==========================================================
# ⚡ WEPS ReflexNeuron — Production Grade High-Frequency Logging
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import numpy as np
import logging
import threading
import queue
import time

logger = logging.getLogger("WEPS.Neurons.ReflexNeuron")

class ReflexNeuron:
    """
    WEPS ReflexNeuron
    - High-frequency logging optimized for production.
    - Async log queue with background thread (optional).
    - Verbose flag for debug logs.
    """
    def __init__(self, state_outputs: dict, phase: str = "neutral", verbose: bool = False):
        self.state_outputs = state_outputs
        self.phase = phase
        self.verbose = verbose

        # Async logging queue setup (optional)
        self._log_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._log_thread = threading.Thread(target=self._process_log_queue, daemon=True)
        self._log_thread.start()

        if self.verbose:
            self._enqueue_log(logging.DEBUG, f"ReflexNeuron initialized with phase={self.phase}")

    def compute(self) -> dict:
        try:
            phase_shift_conf = self._safe_get("phase_transition", "phase_shift_confidence")
            elliott_bias = 1 if self._safe_get("elliott_wave", "impulse_strength", fallback=0) > 0 else -1
            momentum = np.clip(self._safe_get("momentum", "momentum_score_norm"), -1, 1)
            sentiment_bias = 1 if self._safe_get("sentiment", "risk_on_bias", fallback=0) > 0 else -1
            destruction_idx = self._safe_get("immune_response", "destruction_index", fallback=0.0)

            combined = phase_shift_conf * (elliott_bias + momentum + sentiment_bias) / 3.0
            reflex_confidence = np.clip(combined, -1, 1)

            early_exit = destruction_idx > 0.75
            if early_exit:
                reflex_confidence = 0.0

            result = {
                "reflex_confidence": round(reflex_confidence, 4),
                "bias_direction": "buy" if reflex_confidence > 0.2 else "sell" if reflex_confidence < -0.2 else "hold",
                "early_exit_signal": early_exit
            }

            if self.verbose:
                self._enqueue_log(logging.INFO, f"ReflexNeuron compute completed: {result}")

            return result

        except Exception as ex:
            self._enqueue_log(logging.ERROR, f"ReflexNeuron compute error: {ex}")
            return {}

    def _safe_get(self, neuron: str, key: str, fallback=0.0):
        try:
            return self.state_outputs.get(neuron, {}).get(key, fallback)
        except Exception:
            return fallback

    def _enqueue_log(self, level, msg):
        try:
            self._log_queue.put((level, msg))
        except Exception:
            pass  # fail silently to avoid blocking

    def _process_log_queue(self):
        while not self._stop_event.is_set():
            try:
                level, msg = self._log_queue.get(timeout=0.1)
                logger.log(level, msg)
                self._log_queue.task_done()
            except queue.Empty:
                pass  # no log to process, loop

    def shutdown(self):
        self._stop_event.set()
        self._log_thread.join(timeout=1)

