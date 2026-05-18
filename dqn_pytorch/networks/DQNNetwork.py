import torch
from typing import Type
from dqn_pytorch.encoders import NatureDQNEncoder


class DQNNetwork(torch.nn.Module):
    def __init__(
        self,
        action_space: int,
        embedding_dimension: int,
        activation: Type[torch.nn.Module],
    ):
        super().__init__()
        self.phi = NatureDQNEncoder(activation=activation)
        self.q_stream = torch.nn.Sequential(
            torch.nn.Linear(3136, embedding_dimension),
            activation(),
            torch.nn.Linear(embedding_dimension, action_space),
        )

    def forward(self, x):
        features = self.features(x / 255.0)
        return self.q_stream(features)
