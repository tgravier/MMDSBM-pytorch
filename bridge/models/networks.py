import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# === Time Encoding (sinusoidal, non-learned) ===
class TimeEncoding(nn.Module):
    def __init__(self, time_dim: int, max_time: float = 1.0):
        super().__init__()
        self.time_dim = time_dim
        self.max_time = max_time
        self.freqs = torch.exp(
            torch.arange(0, time_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / time_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_norm = t / self.max_time
        angles = t_norm * self.freqs.to(t.device)
        emb = torch.zeros(t.size(0), self.time_dim, device=t.device)
        emb[:, 0::2] = torch.sin(angles)
        emb[:, 1::2] = torch.cos(angles)
        return emb


# === MLP with time projected differently at each layer ===
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        time_dim: int,
        layers_widths: List[int] = [100, 100, 2],
        activation_fn=F.tanh,
    ):
        super().__init__()

        self.activation_fn = activation_fn
        self.layers = nn.ModuleList()
        self.time_projections = nn.ModuleList()
        self.time_dim = time_dim

        prev_width = input_dim
        for layer_width in layers_widths:
            # Linear layer for input
            self.layers.append(nn.Linear(prev_width + layer_width, layer_width))
            # Linear layer for time embedding projection (to match layer_width)
            self.time_projections.append(nn.Linear(time_dim, layer_width))
            prev_width = layer_width

    def forward(self, x: torch.Tensor, t_encoded: torch.Tensor) -> torch.Tensor:
        for i, (layer, time_proj) in enumerate(
            zip(self.layers[:-1], self.time_projections[:-1])
        ):
            t_proj = time_proj(t_encoded)
            x = torch.cat([x, t_proj], dim=-1)
            x = self.activation_fn(layer(x))
        # Dernière couche (output)
        t_proj = self.time_projections[-1](t_encoded)
        x = torch.cat([x, t_proj], dim=-1)
        return self.layers[-1](x)


# === Score Network with Time Encoding ===
class ScoreNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layers_widths: List[int],
        activation_fn=F.tanh,
        time_dim: int = 16,
        max_time: float = 1.0,
    ):
        super().__init__()
        self.time_encoder = TimeEncoding(time_dim=time_dim, max_time=max_time)
        self.net = MLP(
            input_dim=input_dim,
            time_dim=time_dim,
            layers_widths=layers_widths,
            activation_fn=activation_fn,
        )

    def forward(self, x_input: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_encoded = self.time_encoder(t)
        return self.net(x_input, t_encoded)

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, time_dim: int, activation_fn=F.tanh):
        super().__init__()
        self.activation_fn = activation_fn
        self.linear = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)

    def forward(self, x: torch.Tensor, t_encoded: torch.Tensor):
        t_proj = self.time_proj(t_encoded)
        out = torch.cat([x, t_proj], dim=-1)
        out = self.activation_fn(self.linear(out))
        return x + out  # Connexion résiduelle

class ResNetMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        time_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        activation_fn=F.tanh,
        output_dim: int = 2,
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, time_dim, activation_fn)
            for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, t_encoded: torch.Tensor):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x, t_encoded)
        return self.output_layer(x)

class ScoreNetworkResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        activation_fn=F.silu,
        time_dim: int = 16,
        max_time: float = 1.0,
        output_dim: int = 2,
    ):
        super().__init__()
        self.time_encoder = TimeEncoding(time_dim=time_dim, max_time=max_time)
        self.net = ResNetMLP(
            input_dim=input_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            activation_fn=activation_fn,
            output_dim=output_dim,
        )

    def forward(self, x_input: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_encoded = self.time_encoder(t)
        return self.net(x_input, t_encoded)
