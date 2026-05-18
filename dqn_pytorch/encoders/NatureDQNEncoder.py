import torch
from typing import Type


class NatureDQNEncoder(torch.nn.Module):
    def __init__(self, activation: Type[torch.nn.Module] = torch.nn.ReLU):
        self.stream = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 8, stride=4),
            activation(),
            torch.nn.Conv2d(32, 64, 4, stride=2),
            activation(),
            torch.nn.Conv2d(64, 64, 3, stride=1),
            activation(),
            torch.nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stream(x)
