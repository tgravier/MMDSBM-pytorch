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
        experiment_name="tiny_gaussian_01",
        dim=2,
        n_distributions=3,
        first_coupling="ref",
        sigma=1,
        num_simulation_steps=80,
        nb_inner_opt_steps=20000,
        nb_outer_iterations=20,
        eps=1e-3,
        batch_size=64,  #TODO change batchsize its for debug
        lr=1e-4,
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
        time=0, mean=[0]*2, std=[1]*2, n_samples=2000, dim=2
    )

    # Droite
    gaussian2 = GaussianConfig(
        time=1, mean=[5]*2, std=[1]*2, n_samples=2000, dim=2
    )

    # Bas
    gaussian3 = GaussianConfig(
        time=2, mean=[0]*2, std=[0.001]*2, n_samples=2000, dim=2
    )

    # # Droite
    # gaussian4 = GaussianConfig(
    #     time=3, mean=[5, 5.0], std=[1, 1], n_samples=2000, dim=2
    # )


    # # Haut
    # gaussian3 = GaussianConfig(
    #     time=2, mean=[5.0, 9.0], std=[0.5, 0.5], n_samples=2000, dim=2
    # )

    # # Gauche
    # gaussian4 = GaussianConfig(
    #     time=3, mean=[1.0, 5.0], std=[0.5, 0.5], n_samples=2000, dim=2
    # )


    # # Création de cercles pour former un chemin fermé (le dernier revient sur le premier)
    # circle1 = CircleConfig(time=0, center=[0, 0], radius=1.0, n_samples=2000)
    # circle2 = CircleConfig(time=1, center=[2, 2], radius=1.0, n_samples=2000)
    # circle3 = CircleConfig(time=2, center=[4, 4], radius=1.0, n_samples=2000)
    # circle4 = CircleConfig(time=3, center=[2, 6], radius=1.0, n_samples=2000)
    # circle5 = CircleConfig(time=4, center=[0, 0], radius=1.0, n_samples=2000)  # Retour au point de départ

    # Remplace les gaussiennes par les cercles pour l'entraînement
    #distributions_train = [circle1, circle2, circle3, circle4, circle5]

    #distributions_train = [gaussian1, gaussian2, gaussian3, gaussian4]
    distributions_train = [gaussian1, gaussian2, gaussian3]
    max_time = max(distribution.time for distribution in distributions_train)

    print(f"max time:{max_time}")

    net_fwd = ScoreNetwork(
        input_dim=args.dim,
        layers_widths=[128,128, args.dim],
        activation_fn=nn.SiLU(),
        time_dim=64,
        max_time=max_time,
    )

    net_bwd = ScoreNetwork(
        input_dim=args.dim,
        layers_widths=[128,128, args.dim],
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

    trainer.train()


if __name__ == "__main__":
    main()
