#!/usr/bin/env python3
# ==========================================================
# 🧠 WEPS State Sequence Buffer — Spiral Intelligence Edition
# ✅ Encodes categorical phases, states, and mutation signals
# ==========================================================
from collections import deque
import torch

class StateSequenceBuffer:
    """
    WEPS State Sequence Buffer
    Maintains a rolling window of final state vectors (numeric only) for reflex cortex input.
    """

    def __init__(self, seq_len: int, state_keys: list, device: str = 'cpu'):
        """
        Args:
            seq_len: Number of timesteps to keep.
            state_keys: Ordered keys expected in the state dict.
            device: torch device ('cpu' or 'cuda').
        """
        self.seq_len = seq_len
        self.state_keys = state_keys
        self.device = device
        self.buffer = deque(maxlen=seq_len)

    def append(self, state_dict):
        """Append new state vector dict with robust encoding."""
        state_array = []
        for key in self.state_keys:
            value = state_dict.get(key, None)

            if isinstance(value, (float, int)) and value is not None:
                state_array.append(float(value))
            elif isinstance(value, bool):
                state_array.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                encoded = self._encode_categorical(key, value)
                state_array.append(encoded)
            else:
                state_array.append(0.0)  # robust fallback for None or unexpected types

        self.buffer.append(torch.tensor(state_array, dtype=torch.float32))

    def _encode_categorical(self, key, value):
        """Encodes string fields (phase, state) to numeric values."""
        if key == "phase_name":
            mapping = {"rebirth": 1.0, "growth": 2.0, "decay": 3.0}
            return mapping.get(value.lower(), 0.0)
        if key == "alligator_state":
            mapping = {"trend": 1.0, "trend_reverse": 2.0, "range": 3.0}
            return mapping.get(value.lower(), 0.0)
        if key == "risk_state":
            mapping = {"neutral": 0.0, "on": 1.0, "off": -1.0}
            return mapping.get(value.lower(), 0.0)
        return 0.0

    def is_ready(self) -> bool:
        """Returns True if buffer has enough sequences."""
        return len(self.buffer) == self.seq_len

    def get_sequence(self) -> torch.Tensor:
        """Returns current buffer as [1, seq_len, state_dim] tensor."""
        if not self.is_ready():
            raise ValueError(f"Buffer incomplete: {len(self.buffer)}/{self.seq_len} elements.")
        seq = torch.stack(list(self.buffer), dim=0)  # [seq_len, state_dim]
        return seq.unsqueeze(0).to(self.device)

    def reset(self):
        """Clears buffer contents."""
        self.buffer.clear()
