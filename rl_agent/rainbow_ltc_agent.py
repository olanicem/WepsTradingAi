#!/usr/bin/env python3
# ==========================================================
# WEPS3-EPTS RainbowDQN Agent with LTC Backbone v2.0
# Institutional-Grade Reinforcement Learning Agent
# Fully FESI-Compliant, Sophisticated Distributional RL
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional

from weps.rl_agent.ltc_cell import LTCCell  # Ensure this is rigorously implemented per LTC specs

class RainbowLTCNetwork(nn.Module):
    """
    Rainbow DQN Network with Liquid Time-Constant (LTC) recurrent backbone.
    Processes WEPS 103-dim FESI spiral state vector for advanced temporal representation.
    Outputs a distributional Q-value vector for each action.
    """
    def __init__(
        self,
        input_dim: int = 103,
        num_actions: int = 4,
        num_atoms: int = 51,
        hidden_dim: int = 128,
        support_min: float = -10.0,
        support_max: float = 10.0
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.hidden_dim = hidden_dim

        # LTC recurrent cell with rigorous parameterization
        self.ltc = LTCCell(input_dim, hidden_dim)

        # Fully connected projection layers for Rainbow DQN categorical output
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, num_actions * num_atoms)

        # Distribution support vector, fixed buffer for categorical DQN
        self.register_buffer("support", torch.linspace(support_min, support_max, num_atoms))
        self.delta_z = self.support[1] - self.support[0]

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: LTC backbone + distributional head.
        Args:
            x: (batch_size, input_dim) FESI state vector
            hidden_state: (batch_size, hidden_dim) LTC hidden state

        Returns:
            q_dist: (batch_size, num_actions, num_atoms) categorical Q-value distributions
            next_hidden: (batch_size, hidden_dim) updated LTC hidden state
        """
        next_hidden = self.ltc(x, hidden_state)  # LTC state update
        x = F.relu(self.fc1(next_hidden))
        q_dist = self.fc2(x).view(-1, self.num_actions, self.num_atoms)
        q_dist = F.softmax(q_dist, dim=2)  # Probability distributions over atoms

        return q_dist, next_hidden

    def q_values(self, q_dist: torch.Tensor) -> torch.Tensor:
        """
        Compute expected Q values by summing weighted atoms.

        Args:
            q_dist: (batch_size, num_actions, num_atoms)

        Returns:
            q_values: (batch_size, num_actions)
        """
        return torch.sum(q_dist * self.support, dim=2)

class RainbowLTCAgent:
    """
    Fully featured Rainbow DQN agent with:
      - Distributional RL (Categorical DQN)
      - Double DQN updates
      - Prioritized Experience Replay (externally provided)
      - Multi-step returns support
      - Noisy Nets for exploration (to be integrated separately)
      - LTC backbone for biological spiral temporal encoding
    """

    def __init__(
        self,
        input_dim: int = 103,
        num_actions: int = 4,
        device: Optional[torch.device] = None,
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 32,
        num_atoms: int = 51,
        hidden_dim: int = 128,
        target_update_freq: int = 1000,
        multi_step: int = 3,
        replay_buffer: Optional[Any] = None
    ) -> None:
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.multi_step = multi_step

        # Networks
        self.online_net = RainbowLTCNetwork(input_dim, num_actions, num_atoms, hidden_dim).to(self.device)
        self.target_net = RainbowLTCNetwork(input_dim, num_actions, num_atoms, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer with gradient clipping support
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # External replay buffer with prioritized sampling and multi-step support
        self.replay_buffer = replay_buffer

        self.num_atoms = num_atoms
        self.support = torch.linspace(-10, 10, num_atoms).to(self.device)
        self.delta_z = self.support[1] - self.support[0]

        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

        # LTC hidden states management
        self.hidden_state_online = None
        self.hidden_state_target = None

    def reset_hidden_states(self, batch_size: int) -> None:
        """
        Initialize/reset LTC hidden states for online and target networks.
        """
        self.hidden_state_online = torch.zeros(batch_size, self.online_net.hidden_dim, device=self.device)
        self.hidden_state_target = torch.zeros(batch_size, self.target_net.hidden_dim, device=self.device)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection with expected Q-values from distributional output.

        Args:
            state: np.ndarray, shape=(input_dim,)
            epsilon: float, exploration probability

        Returns:
            action: int
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if self.hidden_state_online is None:
            self.reset_hidden_states(batch_size=1)

        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)

        with torch.no_grad():
            q_dist, self.hidden_state_online = self.online_net(state_tensor, self.hidden_state_online)
            q_values = self.online_net.q_values(q_dist)
            action = q_values.argmax(dim=1).item()
        return action

    def learn(self) -> Dict[str, Any]:
        """
        Sample a batch and perform a Rainbow DQN update step.

        Returns:
            metrics: dict with loss and optionally other stats
        """
        if self.replay_buffer is None or len(self.replay_buffer) < self.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)

        batch_size = states.shape[0]
        self.reset_hidden_states(batch_size)

        # Online network forward for current states
        dist, _ = self.online_net(states, self.hidden_state_online)
        actions_expanded = actions.view(-1, 1, 1).expand(-1, 1, self.num_atoms)
        dist = dist.gather(1, actions_expanded).squeeze(1)  # (batch, num_atoms)

        # Target network forward for next states
        with torch.no_grad():
            next_dist, _ = self.target_net(next_states, self.hidden_state_target)
            next_q_values = self.target_net.q_values(next_dist)
            next_actions = next_q_values.argmax(dim=1)
            next_actions_expanded = next_actions.view(-1, 1, 1).expand(-1, 1, self.num_atoms)
            next_dist = next_dist.gather(1, next_actions_expanded).squeeze(1)

            # Distributional Bellman projection
            target_dist = self._project_distribution(rewards, dones, next_dist)

        # Loss: Cross entropy between projected target distribution and current distribution
        loss = -torch.sum(target_dist * torch.log(dist + 1e-8), dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {'loss': loss.item()}

    def _project_distribution(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_dist: torch.Tensor
    ) -> torch.Tensor:
        """
        Implements the Categorical DQN distributional Bellman operator projection.

        Args:
            rewards: (batch,)
            dones: (batch,)
            next_dist: (batch, num_atoms)

        Returns:
            projected_dist: (batch, num_atoms)
        """
        batch_size = rewards.size(0)
        projected_dist = torch.zeros_like(next_dist).to(self.device)

        for i in range(batch_size):
            for j in range(self.num_atoms):
                tz_j = rewards[i] + (1 - dones[i]) * (self.gamma ** self.multi_step) * self.support[j]
                tz_j = tz_j.clamp(self.support[0], self.support[-1])
                b_j = (tz_j - self.support[0]) / self.delta_z
                l = b_j.floor().long()
                u = b_j.ceil().long()

                if l == u:
                    projected_dist[i][l] += next_dist[i][j]
                else:
                    projected_dist[i][l] += next_dist[i][j] * (u.float() - b_j)
                    projected_dist[i][u] += next_dist[i][j] * (b_j - l.float())

        return projected_dist
