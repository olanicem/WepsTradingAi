#!/usr/bin/env python3
# ==============================================================
# ⚡ WEPS Reflex Cortex RL — World’s Most Advanced Trading Brain
# ✅ Spiral-Aware Reflex + Rainbow DQN + Genome-Aware Sentiment + Lifelong Memory
# ✅ Bulletproof next_state validation for production reliability
# Author: Ola Bode (WEPS Creator)
# ==============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import logging

from weps.core.spiral_trade_targets import compute_spiral_trade_targets
from weps.neurons.sentiment_neuron import SentimentNeuron

logger = logging.getLogger("WEPS.ReflexCortexRL")

ACTION_LABELS = ["buy", "sell", "hold"]

class ReflexCortexRL(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=3, replay_capacity=100_000):
        super(ReflexCortexRL, self).__init__()
        self.online_net = RainbowDQN(input_dim, hidden_dim, output_dim)
        self.target_net = RainbowDQN(input_dim, hidden_dim, output_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.spiral_replay = SpiralReplayBuffer(replay_capacity)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-4)
        self.last_action_index = None

        # Initialize with default SentimentNeuron; organism will override in forward()
        self.sentiment_neuron = None

    def forward(self, reflex_state_vector, spiral_meta):
        """
        Makes reflex decision, computes spiral-aware targets with Fib, logs memory.
        Integrates genome-aware sentiment for final trade command.
        """
        state = torch.from_numpy(reflex_state_vector).float().unsqueeze(0)
        q_values = self.online_net(state)
        probs = F.softmax(q_values, dim=-1).detach().numpy().flatten()

        if random.random() < self.epsilon:
            action_idx = random.randint(0, 2)
        else:
            action_idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        self.last_action_index = action_idx

        # Prepare Spiral and Fib data
        fib_levels = spiral_meta.get('fib_levels', {})
        spiral_targets = compute_spiral_trade_targets(
            spiral_meta['entry_base'], spiral_meta['atr'],
            spiral_meta['sei_slope'], spiral_meta['entropy_slope'],
            spiral_meta['spiral_depth'], direction=1 if action_idx == 0 else -1,
            fib_0_382=fib_levels.get("0.382", 0.0),
            fib_0_618=fib_levels.get("0.618", 0.0),
            fib_1_618=fib_levels.get("1.618", 0.0)
        )

        # === Integrate genome-aware sentiment ===
        organism = spiral_meta.get("organism", "UNKNOWN")
        if not self.sentiment_neuron or getattr(self.sentiment_neuron, "organism", None) != organism:
            self.sentiment_neuron = SentimentNeuron(asset_keywords=[organism])
            self.sentiment_neuron.organism = organism  # Track for reuse

        elliott_result = {"direction": spiral_meta.get("elliott_direction", "unknown")}
        phase = spiral_meta.get("phase", "neutral")
        sentiment_output = self.sentiment_neuron.compute(elliott_result, phase)

        # Adjust confidence with sentiment
        adjusted_confidence = max(0.0, min(1.0, confidence + sentiment_output["sentiment_adjustment"]))

        # Build final trade command including sentiment
        trade_command = {
            "action": ACTION_LABELS[action_idx],
            "confidence": round(adjusted_confidence, 5),
            "probability_distribution": probs.tolist(),
            "fib_levels": fib_levels,
            "sentiment_signal": sentiment_output["sentiment_signal"],
            "sentiment_adjustment": sentiment_output["sentiment_adjustment"],
            "average_polarity": sentiment_output["average_polarity"],
            "phase": phase,
            "elliott_direction": elliott_result["direction"],
            **spiral_targets
        }
        logger.info("[ReflexCortexRL] Trade Command with Fibs and Sentiment: %s", trade_command)
        return trade_command

    def store_transition(self, state, action, reward, next_state, done):
        next_state = self._ensure_numeric(next_state, fallback=state)
        self.spiral_replay.add(state, action, reward, next_state, done)

    def train_step(self, batch_size=64):
        if len(self.spiral_replay) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.spiral_replay.sample(batch_size)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        entropy_norm = np.clip(abs(np.mean([s[-1] for s in states.tolist()])), 0.0, 1.0)
        self.epsilon = max(self.epsilon_min, self.epsilon * (self.epsilon_decay ** (1.0 - entropy_norm)))

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _ensure_numeric(self, x, fallback=None):
        """
        Ensures next_state is a numeric np.ndarray compatible with torch.FloatTensor.
        Falls back to current state if None is detected.
        """
        if x is None:
            logger.warning("[ReflexCortexRL] Received None next_state; using fallback state.")
            x = np.copy(fallback)
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)
        if x.dtype.kind not in ('f', 'i'):
            raise ValueError(f"[ReflexCortexRL] Invalid next_state type: {type(x)} with dtype {x.dtype}")
        return x

class RainbowDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RainbowDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        features = self.feature(x)
        advantages = self.advantage(features)
        value = self.value(features)
        return value + (advantages - advantages.mean())

class SpiralReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
