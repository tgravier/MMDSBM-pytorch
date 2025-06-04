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
from datasets.datasets_registry import GaussianConfig
    




def main():
    args = SimpleNamespace(
        seed=42,
        experiment_dir = "experiments_debug",
        experiment_name = 'test_bouncing_3_2bridge_gobackdistrib',
        dim=2,
        n_distributions=3,
        first_coupling ='ref',
        sigma=1.0,
        num_simulation_steps=8,
        nb_inner_opt_steps=10000,
        nb_outer_iterations=20,
        eps = 1e-3,
        batch_size=128,
        lr = 1e-4,
        grad_clip=1.0,
        vis_every=1,
        accelerator= Accelerator()
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Using device: {args.accelerator.device}")

    # Instanciation de deux gaussiennes

    # Gaussienne centrée en (0, 0), std = 1 pour chaque dimension
    gaussian1 = GaussianConfig(time=0, mean=[0.0, 0.0], std=[1.0, 1.0], n_samples=1000, dim=2)

    # Gaussienne centrée en (3, 3), std = 1 pour chaque dimension
    gaussian2 = GaussianConfig(time=1, mean=[3.0, 3.0], std=[1.0, 1.0], n_samples=1000, dim=2)

    gaussian3 = GaussianConfig(time = 2, mean = [-2, -2], std = [1.0, 1.0], n_samples=1000, dim=2)

    distributions_train = [gaussian1,gaussian2,gaussian3]
    #distributions_train = [gaussian1, gaussian2]

    # Score networks for forward and backward processes
    net_fwd = ScoreNetwork(input_dim=args.dim + 1, layers_widths=[128,  128, args.dim], activation_fn=nn.SiLU())
    net_bwd = ScoreNetwork(input_dim=args.dim + 1, layers_widths=[128,  128, args.dim], activation_fn=nn.SiLU())

    optimizer = {
    "forward": torch.optim.Adam(net_fwd.parameters(), lr=args.lr),
    "backward": torch.optim.Adam(net_bwd.parameters(), lr=args.lr)
    }

  


    # Schrödinger bridge trainer
    trainer = N_Bridges(
        args = args,
        net_fwd=net_fwd,
        net_bwd=net_bwd,
        optimizer = optimizer,
        n_distribution=args.n_distributions,
        distributions_train=distributions_train
    )

    trainer.train("bouncing")


if __name__ == "__main__":
    main()
