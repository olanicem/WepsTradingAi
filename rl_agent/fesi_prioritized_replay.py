#!/usr/bin/env python3
# ==========================================================
# WEPS3-EPTS Prioritized Experience Replay Buffer v1.0
# Institutional-Grade, High-Performance Replay Buffer for RL Training
# Fully FESI Spiral-Intelligence Compliant
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import numpy as np
import random
from typing import List, Dict, Any, Tuple

class SumTree:
    """
    SumTree data structure supporting:
    - Efficient O(log N) updates and sampling
    - High-throughput priority updates
    - Memory-efficient storage for large-scale replay buffers
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx: int, priority: float) -> None:
        assert priority >= 0, "Priority must be non-negative"
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority: float, data: Any) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def get(self, s: float) -> Tuple[int, float, Any]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total_priority(self) -> float:
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer:
    - Supports multi-step returns for temporal credit assignment
    - Applies importance sampling corrections to reduce bias
    - Efficient prioritized sampling via SumTree
    - Fully aligned with FESI Spiral Intelligence for market lifecycle complexity
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        multi_step: int = 3,
        gamma: float = 0.99,
    ):
        assert capacity > 0, "Capacity must be positive"
        assert 0 <= alpha <= 1, "Alpha must be in [0,1]"
        assert 0 <= beta_start <= 1, "Beta start must be in [0,1]"
        assert multi_step >= 1, "Multi-step must be >= 1"
        assert 0 <= gamma <= 1, "Gamma must be in [0,1]"

        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.frame = 1
        self.multi_step = multi_step
        self.gamma = gamma

        self.sum_tree = SumTree(capacity)
        self.n_step_buffer: List[Dict[str, Any]] = []

    def _get_priority(self, error: float) -> float:
        """Compute priority from TD error with small epsilon for stability."""
        return (abs(error) + 1e-6) ** self.alpha

    def beta_by_frame(self, frame_idx: int) -> float:
        """Anneal beta linearly from beta_start to 1 over beta_frames."""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, transition: Dict[str, Any]) -> None:
        """
        Add a transition with multi-step return aggregation.
        Delays actual storage until enough steps collected.
        """
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.multi_step:
            return

        reward, next_state, done = self._calc_multi_step_return()
        multi_step_transition = {
            'states': self.n_step_buffer[0]['states'],
            'actions': self.n_step_buffer[0]['actions'],
            'rewards': reward,
            'next_states': next_state,
            'dones': done,
        }

        max_priority = np.max(self.sum_tree.tree[-self.sum_tree.capacity:])
        max_priority = max_priority if max_priority > 0 else 1.0

        self.sum_tree.add(max_priority, multi_step_transition)
        self.n_step_buffer.pop(0)

    def _calc_multi_step_return(self) -> Tuple[float, Any, bool]:
        """Compute discounted multi-step reward, next_state, and done flag."""
        reward, next_state, done = 0.0, None, False
        for idx, transition in enumerate(self.n_step_buffer):
            reward += (self.gamma ** idx) * transition['rewards']
            next_state = transition['next_states']
            done = transition['dones']
            if done:
                break
        return reward, next_state, done

    def sample(self, batch_size: int) -> Dict[str, Any]:
        """
        Sample a batch of experiences with prioritized sampling.
        Returns batch dict with importance sampling weights.
        """
        assert batch_size > 0, "Batch size must be positive"
        batch = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }

        idxs = []
        segment = self.sum_tree.total_priority / batch_size
        priorities = []
        self.beta = self.beta_by_frame(self.frame)
        self.frame += 1

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.sum_tree.get(s)
            idxs.append(idx)
            priorities.append(priority)
            batch['states'].append(data['states'])
            batch['actions'].append(data['actions'])
            batch['rewards'].append(data['rewards'])
            batch['next_states'].append(data['next_states'])
            batch['dones'].append(data['dones'])

        sampling_probabilities = np.array(priorities) / self.sum_tree.total_priority
        is_weights = np.power(self.sum_tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        batch = {k: np.array(v) for k, v in batch.items()}

        return {
            'idxs': idxs,
            'batch': batch,
            'is_weights': is_weights,
        }

    def update_priorities(self, idxs: List[int], errors: List[float]) -> None:
        """
        Update priorities for sampled transitions post learning step.
        """
        assert len(idxs) == len(errors), "idxs and errors length mismatch"
        for idx, error in zip(idxs, errors):
            priority = self._get_priority(error)
            self.sum_tree.update(idx, priority)

    def __len__(self) -> int:
        return self.sum_tree.n_entries
