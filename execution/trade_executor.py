#!/usr/bin/env python3
# ==========================================================
# 🛠️ WEPS Simulated Trade Executor — Production Ready
# Author: Ola Bode (WEPS Creator)
# Description:
#   Simulates trade execution using spiral intelligence signals
#   Implements realistic trade gating, reward shaping, and state transitions.
# ==========================================================

import logging
import numpy as np
import random
from typing import Dict, Tuple

logger = logging.getLogger("WEPS.TradeExecutor")
logger.setLevel(logging.INFO)


class TradeExecutor:
    def __init__(self):
        self.position = None  # "long", "short", or None
        self.entry_price = None
        self.max_drawdown = 0.05  # 5% max simulated drawdown
        self.current_drawdown = 0.0
        self.trade_count = 0
        self.max_trades_per_episode = 10
        self.episode_done = False

    def reset(self):
        """Reset executor state for new episode."""
        self.position = None
        self.entry_price = None
        self.current_drawdown = 0.0
        self.trade_count = 0
        self.episode_done = False

    def execute_trade(self, decision: Dict, meta: Dict) -> Tuple[float, np.ndarray, bool]:
        """
        Simulate trade execution based on spiral intelligence signals.

        Args:
            decision (dict): {
                "organism": str,
                "final_action": str ("buy"/"sell"/"hold"),
                "confidence": float,
                ...
            }
            meta (dict): {
                "majority_phase": str,
                "avg_z_score": float,
                "phase_distribution": dict,
                ...
            }

        Returns:
            reward (float): Simulated PnL reward for RL agent
            next_state (np.ndarray): Simulated next state vector (shape=40)
            done (bool): True if episode finished, else False
        """
        if self.episode_done:
            logger.info(f"Episode ended, no more trades allowed.")
            return 0.0, np.zeros(40, dtype=np.float32), True

        organism = decision.get("organism", "UNKNOWN")
        action = decision.get("final_action", "hold").lower()
        confidence = decision.get("confidence", 0.0)
        phase = meta.get("majority_phase", "unknown")

        # Gate trades based on spiral phase & confidence
        if phase not in ["growth", "rebirth"]:
            logger.info(f"[{organism}] Phase '{phase}' not favorable for trading. Skipping trade.")
            return 0.0, self._generate_next_state(), False

        if confidence < 0.5:
            logger.info(f"[{organism}] Confidence {confidence:.3f} below threshold. Skipping trade.")
            return 0.0, self._generate_next_state(), False

        reward = 0.0
        done = False

        # Simulate market price (simplified)
        market_price = random.uniform(0.95, 1.05)

        # Trade logic simulation
        if action == "buy":
            if self.position == "long":
                # Already long, simulate holding
                reward = self._simulate_pnl(market_price)
            else:
                # Enter long position
                self.position = "long"
                self.entry_price = market_price
                self.trade_count += 1
                logger.info(f"[{organism}] Entered LONG at {market_price:.4f}")

        elif action == "sell":
            if self.position == "short":
                # Already short, simulate holding
                reward = self._simulate_pnl(market_price)
            else:
                # Enter short position
                self.position = "short"
                self.entry_price = market_price
                self.trade_count += 1
                logger.info(f"[{organism}] Entered SHORT at {market_price:.4f}")

        elif action == "hold":
            # No new position, simulate reward if in position
            if self.position in ["long", "short"]:
                reward = self._simulate_pnl(market_price)
            else:
                reward = 0.0

        # Check for max trades or max drawdown to end episode
        if self.trade_count >= self.max_trades_per_episode:
            done = True
            self.episode_done = True
            logger.info(f"[{organism}] Max trades reached. Ending episode.")

        if abs(self.current_drawdown) >= self.max_drawdown:
            done = True
            self.episode_done = True
            logger.info(f"[{organism}] Max drawdown reached. Ending episode.")

        next_state = self._generate_next_state()

        logger.info(
            f"[{organism}] Action: {action.upper()} | Confidence: {confidence:.4f} | Phase: {phase} | "
            f"Reward: {reward:.4f} | Drawdown: {self.current_drawdown:.4f} | Trades: {self.trade_count}"
        )

        return reward, next_state, done

    def _simulate_pnl(self, market_price: float) -> float:
        """
        Simulate profit or loss based on current position and market price.

        Args:
            market_price (float): Simulated current market price.

        Returns:
            float: PnL reward value.
        """
        if self.position is None or self.entry_price is None:
            return 0.0

        # PnL calculation depending on position type
        if self.position == "long":
            pnl = market_price - self.entry_price
        elif self.position == "short":
            pnl = self.entry_price - market_price
        else:
            pnl = 0.0

        # Apply confidence-weighted random volatility to PnL to simulate market randomness
        volatility = random.uniform(0.95, 1.05)
        pnl *= volatility

        # Update drawdown (simulate risk)
        if pnl < 0:
            self.current_drawdown += abs(pnl)

        return pnl

    def _generate_next_state(self) -> np.ndarray:
        """
        Generate simulated next state vector for the RL agent.

        Returns:
            np.ndarray: Random state vector of shape (40,)
        """
        return np.random.rand(40).astype(np.float32)
