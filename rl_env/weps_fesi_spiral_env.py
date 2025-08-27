#!/usr/bin/env python3
# ==========================================================
# WEPS3-EPTS Spiral Reinforcement Learning Environment v2.0
# Institutional-Grade FESI Biologically Grounded Market Lifecycle Model
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import gym
from gym import spaces
import numpy as np
from typing import Tuple, Dict, Any

class WEPSFESISpiralEnv(gym.Env):
    """
    Institutional WEPS Spiral RL Environment
    
    Observations:
      - 103-dimensional FESI Spiral State Vector including:
          spiral phase probabilities, wave Z-scores, entropy, impulse, mutation,
          sentiment, metabolic and reflex signals.
    
    Actions:
      0: Hold
      1: Buy Long
      2: Sell Short
      3: Exit Position
    
    Reward:
      Multi-factor composite reward incorporating:
        - Realized P&L on trades
        - Wave confidence bonus
        - Entropy-based risk penalty
        - Mutation (regime instability) penalty
        - Spiral phase alignment bonus
    
    Termination Criteria:
      - Spiral phase 'death' reached
      - Max step limit exceeded
    """

    PHASES = ['rebirth', 'growth', 'decay', 'death', 'neutral']

    def __init__(self, state_dim: int = 103, max_steps: int = 1000):
        super().__init__()
        self.state_dim = state_dim
        self.max_steps = max_steps

        # Gym space definitions
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Internal environment state
        self.current_step = 0
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)

        # Trade state tracking
        self.position = None   # 'long', 'short', or None
        self.entry_price = None
        self.trade_open = False

        # Data source placeholders (to be integrated with WEPS pipeline)
        self.price_series = None
        self.price_index = 0

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        """
        self.current_step = 0
        self.position = None
        self.entry_price = None
        self.trade_open = False
        self.price_index = 0

        self.current_state = self._load_initial_state()
        return self.current_state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one timestep within the environment.
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        done = False
        info = {}

        # Simulate next spiral lifecycle and market state
        self.current_state = self._simulate_next_state()
        self.current_step += 1

        # Extract key state components
        price = self._extract_price(self.current_state)
        phase = self._extract_spiral_phase(self.current_state)
        wave_confidence = self._extract_wave_confidence(self.current_state)
        entropy = self._extract_entropy(self.current_state)
        mutation_level = self._extract_mutation(self.current_state)

        # Process agent action and compute immediate reward (e.g. trade P&L)
        reward += self._process_action(action, price)

        # Reward shaping for wave confidence, entropy risk, mutation, phase alignment
        reward += self._reward_wave_confidence(wave_confidence)
        reward += self._penalty_entropy(entropy)
        reward += self._penalty_mutation(mutation_level)
        reward += self._reward_phase_alignment(phase, action)

        # Termination: spiral 'death' phase or max steps
        if phase == 'death' or self.current_step >= self.max_steps:
            done = True

        # Diagnostics info
        info.update({
            'step': self.current_step,
            'position': self.position,
            'price': price,
            'phase': phase,
            'wave_confidence': wave_confidence,
            'entropy': entropy,
            'mutation_level': mutation_level
        })

        return self.current_state, reward, done, info

    # --- Internal helpers ---

    def _load_initial_state(self) -> np.ndarray:
        """
        Load or initialize the initial FESI spiral state vector.
        To integrate with live WEPS pipeline data.
        """
        # Placeholder: Gaussian noise simulating starting vector
        return np.random.normal(0, 1, self.state_dim).astype(np.float32)

    def _simulate_next_state(self) -> np.ndarray:
        """
        Simulate market spiral lifecycle dynamics advancing the state vector.
        - Inject Markovian spiral phase transitions
        - Model entropy/mutation shifts biologically
        To be replaced by real WEPS pipeline updates.
        """
        noise = np.random.normal(0, 0.01, self.state_dim)
        next_state = self.current_state + noise

        # TODO: Insert Markov Spiral Lifecycle logic here

        return next_state.astype(np.float32)

    def _extract_price(self, state: np.ndarray) -> float:
        """
        Extract proxy for price from state vector index 0.
        Scaled by 100 for realistic range.
        """
        return float(state[0] * 100)

    def _extract_spiral_phase(self, state: np.ndarray) -> str:
        """
        Extract spiral phase probabilities from state vector indices 95-99,
        return phase with max probability.
        """
        phase_probs = state[95:100]
        idx = np.argmax(phase_probs)
        return self.PHASES[idx]

    def _extract_wave_confidence(self, state: np.ndarray) -> float:
        """
        Extract wave confidence proxy from index 5 (bounded [0,1]).
        """
        return np.clip(float(state[5]), 0.0, 1.0)

    def _extract_entropy(self, state: np.ndarray) -> float:
        """
        Extract entropy value from index 10.
        """
        return max(0.0, float(state[10]))

    def _extract_mutation(self, state: np.ndarray) -> float:
        """
        Extract mutation/regime instability from index 15.
        """
        return max(0.0, float(state[15]))

    # --- Trade action processing ---

    def _process_action(self, action: int, price: float) -> float:
        """
        Handle trade execution logic and compute realized P&L reward.
        """
        reward = 0.0

        if action == 1:  # Buy Long
            if not self.trade_open:
                self.position = 'long'
                self.entry_price = price
                self.trade_open = True

        elif action == 2:  # Sell Short
            if not self.trade_open:
                self.position = 'short'
                self.entry_price = price
                self.trade_open = True

        elif action == 3:  # Exit
            if self.trade_open:
                reward = self._calculate_pnl(price)
                self.position = None
                self.entry_price = None
                self.trade_open = False

        # Hold action (0) results in no trade, no reward update
        return reward

    def _calculate_pnl(self, exit_price: float) -> float:
        """
        Compute profit and loss for current open position.
        """
        if not self.trade_open or self.entry_price is None:
            return 0.0

        if self.position == 'long':
            return exit_price - self.entry_price

        elif self.position == 'short':
            return self.entry_price - exit_price

        return 0.0

    # --- Reward shaping ---

    def _reward_wave_confidence(self, confidence: float) -> float:
        """
        Positive reward proportional to wave confidence strength.
        """
        return 0.05 * confidence

    def _penalty_entropy(self, entropy: float) -> float:
        """
        Negative penalty for entropy above baseline (0.5).
        """
        excess_entropy = max(0.0, entropy - 0.5)
        return -0.1 * excess_entropy

    def _penalty_mutation(self, mutation: float) -> float:
        """
        Negative penalty proportional to mutation/regime instability.
        """
        return -0.2 * mutation

    def _reward_phase_alignment(self, phase: str, action: int) -> float:
        """
        Reward alignment of action with current spiral phase:
          - Buy in rebirth/growth
          - Sell in decay
          - Exit in death
        """
        reward = 0.0
        if phase in ['rebirth', 'growth'] and action == 1:
            reward += 0.1
        if phase == 'decay' and action == 2:
            reward += 0.1
        if phase == 'death' and action == 3:
            reward += 0.2
        return reward
