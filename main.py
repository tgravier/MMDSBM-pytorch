import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import MultivariateNormal
from accelerate import Accelerator
import numpy as np
from types import SimpleNamespace

from bridge.nbridges import N_Bridges
from bridge.models.networks import ScoreNetwork
from datasets.datasets_registry import GaussianConfig, CircleConfig


def main():
    args = SimpleNamespace(
        seed=42,
        experiment_dir="experiments_debug",
        experiment_name="analysis_score_07_circles",
        method="stochastic2",
        dim=2,
        n_distributions=2,
        first_coupling="ref",
        sigma=1,
        num_simulation_steps=20,
        nb_inner_opt_steps=20000,
        nb_outer_iterations=20,
        eps=1e-3,
        batch_size=128,
        lr=5e-5,
        grad_clip=1.0,
        vis_every=1,
        accelerator=Accelerator(),
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Using device: {args.accelerator.device}")

    # Instanciation de quatre gaussiennes formant un losange élargi

    # Bas
    gaussian1 = GaussianConfig(
        time=0, mean=[5.0, 1.0], std=[1, 1], n_samples=2000, dim=2
    )

    # Droite
    gaussian2 = GaussianConfig(
        time=1, mean=[9.0, 5.0], std=[0.1, 0.1], n_samples=2000, dim=2
    )


   
    # Deux cercles paramétrés
    circle1 = CircleConfig(
        time=0,
        n_samples=2000,
        center=[2.0, 2.0],
        radius=1.0,
        thickness=0.05,
    )

    circle2 = CircleConfig(
        time=1,
        n_samples=2000,
        center=[7.0, 7.0],
        radius=1.5,
        thickness=0.08,
    )


    # Haut
    gaussian3 = GaussianConfig(
        time=2, mean=[5.0, 9.0], std=[0.5, 0.5], n_samples=2000, dim=2
    )

    # Gauche
    gaussian4 = GaussianConfig(
        time=3, mean=[1.0, 5.0], std=[0.5, 0.5], n_samples=2000, dim=2
    )

    gaussian4 = GaussianConfig(time = 3, mean = [0, 4.0], std = [1, 1], n_samples=2000, dim=2)

    #distributions_train = [gaussian1, gaussian2, gaussian3, gaussian4]
    distributions_train = [circle1, circle2]
    max_time = max(distribution.time for distribution in distributions_train)

    print(f"max time:{max_time}")

    net_fwd = ScoreNetwork(
        input_dim=args.dim,
        layers_widths=[256, 256, 256, 256, args.dim],
        activation_fn=nn.SiLU(),
        time_dim=16,
        max_time=max_time,
    )

    net_bwd = ScoreNetwork(
        input_dim=args.dim,
        layers_widths=[256, 256, 256, 256, args.dim],
        activation_fn=nn.SiLU(),
        time_dim=64,
        max_time=max_time,
    )

    optimizer = {
        "forward": torch.optim.Adam(net_fwd.parameters(), lr=args.lr),
        "backward": torch.optim.Adam(net_bwd.parameters(), lr=args.lr),
    }

    # Schrödinger bridge trainer
    trainer = N_Bridges(
        args=args,
        net_fwd=net_fwd,
        net_bwd=net_bwd,
        optimizer=optimizer,
        n_distribution=args.n_distributions,
        distributions_train=distributions_train,
    )

    trainer.train(args.method)


if __name__ == "__main__":
    main()
