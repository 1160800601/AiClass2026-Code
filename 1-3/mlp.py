import torch
from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        input_dim = 40
        hidden_dim = 192
        hidden_layers = 5

        layers = []
        dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
