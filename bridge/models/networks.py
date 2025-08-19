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


import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Gaussian Fourier time embedding (classique pour diffusion) ---
class GaussianFourierTimeEncoding(nn.Module):
    def __init__(self, time_dim: int, max_time: float, sigma: float = 1.0, learnable: bool = False):
        """
        time_dim: dimension de l'embedding final (doit être pair).
        max_time: t sera normalisé par max_time.
        sigma: écart-type des fréquences gaussiennes.
        """
        super().__init__()
        assert time_dim % 2 == 0, "time_dim doit être pair (sin/cos)."
        self.time_dim = time_dim
        self.max_time = max_time

        # fréquences ~ N(0, sigma^2)
        freqs = torch.randn(time_dim // 2) * sigma
        if learnable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) ou (B,1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B,1)
        t = t.to(dtype=torch.float32)
        t_norm = t / self.max_time
        angles = 2 * math.pi * t_norm * self.freqs.view(1, -1)  # (B, D/2)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, D)
        return emb


# --- Petit MLP pour convertir l'embedding temps en gammas/betas (FiLM) ---
class TimeToFiLM(nn.Module):
    def __init__(self, time_dim: int, layer_widths: List[int]):
        super().__init__()
        self.layer_widths = layer_widths
        out_dim = 2 * sum(layer_widths)  # gamma & beta par couche
        hidden = max(128, time_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, t_emb: torch.Tensor):
        fb = self.net(t_emb)  # (B, 2*sum(widths))
        gammas, betas = [], []
        idx = 0
        for w in self.layer_widths:
            g = fb[:, idx : idx + w]
            b = fb[:, idx + w : idx + 2 * w]
            gammas.append(g)
            betas.append(b)
            idx += 2 * w
        return gammas, betas  # listes de (B, w)


# --- Bloc MLP + FiLM, résiduel si possible ---
class FiLMBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation_fn=F.silu):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.activation_fn = activation_fn
        self.can_residual = (in_dim == out_dim)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        # FiLM: scale & shift par (1 + gamma) pour stabilité
        h = h * (1.0 + gamma) + beta
        h = self.activation_fn(h)
        if self.can_residual:
            h = h + x
        return h


# --- Réseau de score "classique diffusion" avec API conservée ---
class ScoreNetworkFILM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layers_widths: List[int],
        activation_fn,
        time_dim: int ,
        max_time: int,
    ):
        """
        API conservée:
          - input_dim: dimension de x
          - layers_widths: liste de largeurs cachées; la dernière couche de sortie est fixée à input_dim
          - activation_fn: fonction d'activation (par défaut SiLU)
          - time_dim: dimension de l'embedding temporel
          - max_time: normalisation de t (t/max_time)
        """
        super().__init__()
        self.activation_fn = activation_fn if activation_fn is not None else F.silu

        # Time encoder (Fourier) + convertisseur vers FiLM params
        self.time_encoder = GaussianFourierTimeEncoding(time_dim=time_dim, max_time=max_time)
        self.time_to_film = TimeToFiLM(time_dim=time_dim, layer_widths=layers_widths)

        # Backbone FiLM MLP
        blocks = []
        prev = input_dim
        for w in layers_widths:
            blocks.append(FiLMBlock(prev, w, activation_fn=self.activation_fn))
            prev = w
        self.blocks = nn.ModuleList(blocks)

        # Tête de sortie → score de dim input_dim (init = 0)
        self.out = nn.Linear(prev, input_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_input: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_input: (B, input_dim)
        t: (B,) ou (B,1)
        return: (B, input_dim)
        """
        t_emb = self.time_encoder(t)                # (B, time_dim)
        gammas, betas = self.time_to_film(t_emb)    # listes len=L de (B, w_l)

        h = x_input
        for block, g, b in zip(self.blocks, gammas, betas):
            # adapter (B, w) -> broadcast sur batch
            h = block(h, g, b)

        return self.out(h)


def print_trainable_params(model, name_of_network: str):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    title = f" {name_of_network.strip()} "
    print("=" * ((40 - len(title)) // 2) + title + "=" * ((40 - len(title) + 1) // 2))
    print(f"Trainable parameters : {total_params:,}")
    print("=" * 40)
