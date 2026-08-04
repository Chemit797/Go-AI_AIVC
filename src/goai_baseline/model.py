"""The fixed two-hidden-layer MLP from the document baseline."""

from __future__ import annotations

import torch
from torch import nn


class ConditionMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ResponseDecompositionMLP(nn.Module):
    """Predict a condition background plus a chemical-conditioned response."""

    def __init__(
        self,
        full_input_dim: int,
        background_input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.background_network = nn.Sequential(
            nn.Linear(background_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.response_network = nn.Sequential(
            nn.Linear(full_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def components(self, full_inputs: torch.Tensor, background_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.background_network(background_inputs), self.response_network(full_inputs)

    def forward(self, full_inputs: torch.Tensor, background_inputs: torch.Tensor) -> torch.Tensor:
        background, response = self.components(full_inputs, background_inputs)
        return background + response
