#models/networks

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import List

""" Here we design all the networks for the base case and the 2 marginals case"""




# === Time Encoding (sinusoidal, non-learned) ===
class TimeEncoding(nn.Module):
    def __init__(self, time_dim: int, max_time: float = 1.0):
        super().__init__()
        self.time_dim = time_dim
        self.max_time = max_time

        # Frequency terms for sine/cosine
        self.freqs = torch.exp(
            torch.arange(0, time_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / time_dim)
        )  # shape: [time_dim // 2]

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: Tensor of shape [batch_size, 1] with continuous time values in [0, max_time]
        returns: [batch_size, time_dim]
        """
        
        t_norm = t / self.max_time  # Normalize t to [0, 1]

        angles = t_norm * self.freqs.to(t.device)
        emb = torch.zeros(t.size(0), self.time_dim).to(t.device)
        emb[:, 0::2] = torch.sin(angles)
        emb[:, 1::2] = torch.cos(angles)
        return emb


# === Simple MLP ===
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layers_widths: List[int] = [100, 100, 2],
        activation_fn=F.tanh
    ) -> None:
        super().__init__()

        layers = []
        prev_width = input_dim
        for layer_width in layers_widths:
            layers.append(nn.Linear(prev_width, layer_width))
            prev_width = layer_width

        self.layers = nn.ModuleList(layers)
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        return self.layers[-1](x)


# === Score Network ===
class ScoreNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layers_widths: List[int],
        activation_fn=F.tanh,
        time_dim: int = 16,
        max_time: float = 1.0,
    ) -> None:
        super().__init__()

        self.time_encoder = TimeEncoding(time_dim=time_dim, max_time=max_time) #TODO for the scale encode time for each layers and project on it.

        self.net = MLP(
            input_dim=input_dim + time_dim,
            layers_widths=layers_widths,
            activation_fn=activation_fn
        )

    def forward(self, x_input: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_input: shape [batch_size, dim]
        t: shape [batch_size, 1]
        returns: score estimate
        """
        t_encoded = self.time_encoder(t)  # shape [batch_size, time_dim]
        inputs = torch.cat([x_input, t_encoded], dim=1)
        return self.net(inputs)
