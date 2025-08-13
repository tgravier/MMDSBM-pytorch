import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# === Time Encoding (sinusoidal, non-learned) ===
class TimeEncoding(nn.Module):
    def __init__(self, time_dim: int, max_time):
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
        layers_widths: List[int],
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
        activation_fn,
        time_dim: int,
        max_time: float,
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
    def __init__(self, input_dim: int, output_dim: int, time_dim: int, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Linear layer for input + time projection
        self.linear = nn.Linear(input_dim + output_dim, output_dim)
        # Time projection to match output_dim
        self.time_proj = nn.Linear(time_dim, output_dim)

        # Skip connection: always present, with projection if needed
        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x: torch.Tensor, t_encoded: torch.Tensor):
        t_proj = self.time_proj(t_encoded)
        out = torch.cat([x, t_proj], dim=-1)
        out = self.activation_fn(self.linear(out))
        skip = self.skip_proj(x)
        return skip + out


class ResNetMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        time_dim: int,
        layers_widths: List[int],
        activation_fn=F.tanh,
    ):
        super().__init__()
        self.activation_fn = activation_fn
        self.time_dim = time_dim
        self.layers_widths = layers_widths

        self.blocks = nn.ModuleList()
        prev_width = input_dim
        for layer_width in layers_widths:
            self.blocks.append(
                ResidualBlock(prev_width, layer_width, time_dim, activation_fn)
            )
            prev_width = layer_width

    def forward(self, x: torch.Tensor, t_encoded: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.blocks[:-1]):
            x = block(x, t_encoded)
        # Dernière couche (output) - pas d'activation pour la sortie finale
        return self.blocks[-1](x, t_encoded)


class ScoreNetworkResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layers_widths: List[int],
        activation_fn,
        time_dim: int,
        max_time: float,
    ):
        super().__init__()
        self.time_encoder = TimeEncoding(time_dim=time_dim, max_time=max_time)
        self.net = ResNetMLP(
            input_dim=input_dim,
            time_dim=time_dim,
            layers_widths=layers_widths,
            activation_fn=activation_fn,
        )

    def forward(self, x_input: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_encoded = self.time_encoder(t)
        return self.net(x_input, t_encoded)


def print_trainable_params(model, name_of_network: str):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    title = f" {name_of_network.strip()} "
    print("=" * ((40 - len(title)) // 2) + title + "=" * ((40 - len(title) + 1) // 2))
    print(f"Trainable parameters : {total_params:,}")
    print("=" * 40)
