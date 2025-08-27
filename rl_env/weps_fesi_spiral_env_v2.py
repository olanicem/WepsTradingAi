#!/usr/bin/env python3
# ================================================================
# WEPSFESISpiralEnv — Institutional-Grade RL Environment integrating WEPS Spiral Intelligence
# Author: Ola Bode (WEPS Creator)
# ================================================================

import gym
from gym import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional

from weps.core.weps_pipeline_interface import WEPSPipelineInterface

class WEPSFESISpiralEnv(gym.Env):
    """
    WEPSFESISpiralEnv - Full Institutional FESI-Compliant Spiral RL Environment.

    Observations:
        103-dimensional enriched FESI spiral state vector from WEPS pipeline.

    Actions:
        0: Hold
        1: Buy Long
        2: Sell Short
        3: Exit Position

    Rewards:
        Real-time profit and loss, wave confidence bonus, timing factor bonus,
        sentiment bias adjustment, Fibonacci target proximity bonus,
        entropy and mutation penalties.

    Termination:
        Episode ends if spiral phase is 'death' or max steps reached.
    """

    PHASES = ['rebirth', 'growth', 'decay', 'death', 'neutral']

    def __init__(self, organism: str = "EURUSD", max_steps: int = 1000, timeframes: Optional[list] = None):
        super().__init__()

        self.organism = organism.upper()
        self.max_steps = max_steps
        self.timeframes = timeframes or ["1h", "4h", "1d"]

        # Initialize WEPS pipeline interface for organism/timeframes
        self.pipeline = WEPSPipelineInterface(self.organism, timeframes=self.timeframes)
        self.current_step = 0

        # Gym spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(103,), dtype=np.float32)

        # Trade state tracking
        self.position = None    # 'long', 'short', or None
        self.entry_price = None
        self.trade_open = False

        # Reset pipeline and get initial state
        self.state, _ = self.pipeline.reset()

    def reset(self) -> np.ndarray:
        """Reset environment and pipeline, clear trades, return initial state vector."""
        self.current_step = 0
        self.position = None
        self.entry_price = None
        self.trade_open = False

        self.state, _ = self.pipeline.reset()
        return self.state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        # Advance pipeline one step and update state vector
        self.pipeline.current_step = self.current_step
        self.state, info = self.pipeline.step()

        price = self._get_price()
        phase = self._get_spiral_phase()
        wave_confidence = self._get_wave_confidence()
        entropy = self._get_entropy()
        mutation = self._get_mutation()
        sentiment_score = self._get_sentiment()
        fib_levels = self._get_fib_levels()
        timing_factor_x = self._get_timing_factor()
        atr = self._get_atr()

        reward = 0.0
        realized_pnl = 0.0

        # Step 1: Enforce wave confidence threshold before trade entry
        min_z_threshold = 0.90
        if wave_confidence < min_z_threshold and action in [1, 2]:
            reward -= 0.1
            info['wave_confidence_penalty'] = True

        # Step 2: Position management & P&L calculation
        if action == 1:  # Buy Long
            if not self.trade_open:
                self.position = 'long'
                self.entry_price = price
                self.trade_open = True
                info['trade'] = "Opened Long"

        elif action == 2:  # Sell Short
            if not self.trade_open:
                self.position = 'short'
                self.entry_price = price
                self.trade_open = True
                info['trade'] = "Opened Short"

        elif action == 3:  # Exit
            if self.trade_open and self.entry_price is not None:
                if self.position == 'long':
                    realized_pnl = price - self.entry_price
                else:
                    realized_pnl = self.entry_price - price

                reward += realized_pnl
                info['realized_pnl'] = realized_pnl

                self.position = None
                self.entry_price = None
                self.trade_open = False
                info['trade'] = "Exited Position"

        # Step 3: Reward shaping

        # Phase-aligned trade bonus
        if phase in ['rebirth', 'growth'] and action == 1:
            reward += 0.15
        if phase == 'decay' and action == 2:
            reward += 0.15
        if phase == 'death' and action == 3:
            reward += 0.3

        # Wave confidence bonus (max 10%)
        reward += 0.1 * wave_confidence

        # Timing factor x bonus (corrective wave exhausted)
        if timing_factor_x >= 1 and action in [1, 2]:
            reward += 0.1

        # Sentiment-driven bias adjustment
        if sentiment_score > 0.005:
            if action == 1:
                reward += 0.05
            elif action == 2:
                reward -= 0.05
        elif sentiment_score < -0.005:
            if action == 2:
                reward += 0.05
            elif action == 1:
                reward -= 0.05

        # Fibonacci proximity bonus
        fib_target_price = fib_levels.get('target_Y')
        if fib_target_price is not None and self.trade_open and atr:
            price_distance = abs(price - fib_target_price)
            if price_distance <= 0.5 * atr:
                reward += 0.1
                info['fib_target_hit'] = True

        # Entropy penalty (chaos)
        if entropy > 0.5:
            reward -= 0.15 * (entropy - 0.5)

        # Mutation penalty (instability)
        if mutation > 0.1:
            reward -= 0.2 * mutation

        # Step 4: Check termination
        done = (phase == 'death') or (self.current_step >= self.max_steps)

        # Step 5: Update step count
        self.current_step += 1

        # Step 6: Compose diagnostics info
        info.update({
            "position": self.position,
            "price": price,
            "phase": phase,
            "wave_confidence": wave_confidence,
            "entropy": entropy,
            "mutation": mutation,
            "sentiment_score": sentiment_score,
            "timing_factor_x": timing_factor_x,
            "fib_levels": fib_levels,
            "realized_pnl": realized_pnl,
            "current_step": self.current_step,
            "done": done,
        })

        return self.state, reward, done, info

    # ---- Helper methods to fetch pipeline signals ----

    def _get_price(self) -> float:
        try:
            return float(self.pipeline.pipeline.get_price())
        except Exception:
            return 0.0

    def _get_spiral_phase(self) -> str:
        try:
            return self.pipeline.pipeline.final_spiral_phase or 'neutral'
        except Exception:
            return 'neutral'

    def _get_wave_confidence(self) -> float:
        try:
            return float(self.pipeline.pipeline.spiral_wave_result.get('z_score', 0.0))
        except Exception:
            return 0.0

    def _get_entropy(self) -> float:
        try:
            return float(self.pipeline.pipeline.entropy_norm)
        except Exception:
            return 0.0

    def _get_mutation(self) -> float:
        try:
            return float(self.pipeline.pipeline.mutation_norm if hasattr(self.pipeline.pipeline, 'mutation_norm') else 0.0)
        except Exception:
            return 0.0

    def _get_sentiment(self) -> float:
        try:
            return float(self.pipeline.pipeline.sentiment_score)
        except Exception:
            return 0.0

    def _get_fib_levels(self) -> dict:
        try:
            return self.pipeline.pipeline.spiral_wave_result.get('fib_levels', {})
        except Exception:
            return {}

    def _get_timing_factor(self) -> float:
        try:
            return float(self.pipeline.pipeline.timing_factor_x if hasattr(self.pipeline.pipeline, 'timing_factor_x') else 0.0)
        except Exception:
            return 0.0

    def _get_atr(self) -> Optional[float]:
        try:
            return float(self.pipeline.pipeline.multi_tf_indicator_features.get('atr', 1e-8))
        except Exception:
            return 1e-8
