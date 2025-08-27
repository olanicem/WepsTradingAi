#!/usr/bin/env python3
# ==========================================================
# WEPS3-EPTS Liquid Time-Constant (LTC) Cell v1.1
# Institutional-Grade Continuous-Time Recurrent Cell
# Models neural dynamics reflecting spiral market temporal behavior
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class LTCCell(nn.Module):
    """
    Liquid Time-Constant (LTC) Cell
    
    This cell implements continuous-time recurrent dynamics:
    - Adaptive time constants τ for each neuron enable flexible memory decay and retention
    - Reset gate modulates influence of prior hidden state
    - Differential equation solved via Euler integration with timestep dt
    
    Inputs:
        x: Input tensor of shape [batch_size, input_dim]
        hidden: Hidden state tensor [batch_size, hidden_dim]
        dt: Integration timestep (default=1.0)
        
    Outputs:
        new_hidden: Updated hidden state tensor [batch_size, hidden_dim]
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super(LTCCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Parameter matrices for time constant τ, reset gate r, and candidate state h̃
        self.W_tau = nn.Parameter(torch.Tensor(hidden_dim, input_dim + hidden_dim))
        self.b_tau = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_r = nn.Parameter(torch.Tensor(hidden_dim, input_dim + hidden_dim))
        self.b_r = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_h = nn.Parameter(torch.Tensor(hidden_dim, input_dim + hidden_dim))
        self.b_h = nn.Parameter(torch.Tensor(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights with Kaiming uniform distribution for stable learning
        Initialize biases to zero for unbiased starting point
        """
        nn.init.kaiming_uniform_(self.W_tau, a=5 ** 0.5)
        nn.init.zeros_(self.b_tau)
        nn.init.kaiming_uniform_(self.W_r, a=5 ** 0.5)
        nn.init.zeros_(self.b_r)
        nn.init.kaiming_uniform_(self.W_h, a=5 ** 0.5)
        nn.init.zeros_(self.b_h)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Forward pass computes the next hidden state.

        Parameters:
            x: Current input [batch_size, input_dim]
            hidden: Previous hidden state [batch_size, hidden_dim]
            dt: Integration time step (default=1.0)

        Returns:
            new_hidden: Next hidden state [batch_size, hidden_dim]
        """
        # Concatenate input and hidden state for joint processing
        combined = torch.cat((x, hidden), dim=1)  # [batch_size, input_dim + hidden_dim]

        # Compute adaptive neuron-specific time constants τ (positive and bounded)
        tau_raw = F.linear(combined, self.W_tau, self.b_tau)
        tau = F.softplus(tau_raw) + 1e-6  # Ensure positivity and numerical stability

        # Reset gate controls memory update influence, sigmoid-activated
        r = torch.sigmoid(F.linear(combined, self.W_r, self.b_r))

        # Candidate state h̃ computed with gated hidden state and tanh activation
        gated_hidden = r * hidden
        candidate_input = torch.cat((x, gated_hidden), dim=1)
        h_tilde = torch.tanh(F.linear(candidate_input, self.W_h, self.b_h))

        # Differential state update per continuous-time dynamics: dh/dt = (h̃ - h) / τ
        dh = (h_tilde - hidden) / tau

        # Euler integration step to update hidden state with timestep dt
        new_hidden = hidden + dt * dh

        return new_hidden
