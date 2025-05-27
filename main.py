import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import MultivariateNormal
from accelerate import Accelerator
import numpy as np
from types import SimpleNamespace

from bridge.trainer_dsbm import IMF_DSBM
from bridge.models.networks import ScoreNetwork
    




def main():
    args = SimpleNamespace(
        seed=42,
        experiment_dir = "experiments",
        experiment_name = 'circle_circle_01',
        dim=2,
        first_coupling ='ref',
        sigma=1.0,
        num_simulation_steps=20,
        nb_inner_opt_steps=10000,
        nb_outer_iterations=20,
        eps = 1e-3,
        batch_size=128,
        grad_clip=1.0,
        vis_every=1,
        accelerator= Accelerator()
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Using device: {args.accelerator.device}")


    def sample_ring_circle_to_circle(n_samples, r0_range=(2.0, 2.5), r1_range=(4.0, 4.5)):
        """
        Échantillonne des points appariés dans deux anneaux de rayons différents (r0_range -> r1_range).
        Les points conservent leur angle mais changent de rayon.
        """
        theta = 2 * torch.pi * torch.rand(n_samples)  # même angle pour x0 et x1

        # Rayon initial et final (différents)
        r0 = torch.sqrt(torch.rand(n_samples) * (r0_range[1]**2 - r0_range[0]**2) + r0_range[0]**2)
        r1 = torch.sqrt(torch.rand(n_samples) * (r1_range[1]**2 - r1_range[0]**2) + r1_range[0]**2)

        # x0 : cercle intérieur
        x0 = torch.stack([r0 * torch.cos(theta), r0 * torch.sin(theta)], dim=1)

        # x1 : cercle extérieur (même angle)
        x1 = torch.stack([r1 * torch.cos(theta), r1 * torch.sin(theta)], dim=1)

        return x0, x1



    # Training data (5000 samples)
    x0, x1 = sample_ring_circle_to_circle(n_samples=5000)
    x_pairs = torch.stack([x0, x1], dim=1).to(args.accelerator.device)

    # Test data (2000 samples)
    x0_test, x1_test = sample_ring_circle_to_circle(n_samples=2000)
    x_pairs_test = torch.stack([x0_test, x1_test], dim=1).to(args.accelerator.device)


    # Score networks for forward and backward processes
    net_fwd = ScoreNetwork(input_dim=args.dim + 1, layers_widths=[128, 128, args.dim], activation_fn=nn.SiLU())
    net_bwd = ScoreNetwork(input_dim=args.dim + 1, layers_widths=[128, 128, args.dim], activation_fn=nn.SiLU())

    # Schrödinger bridge trainer
    trainer = IMF_DSBM(
        x_pairs= x_pairs,
        x_pairs_test = x_pairs_test,
        T_min=0.0,
        T_max=1.0,
        args=args,
        num_simulation_steps=args.num_simulation_steps,
        net_fwd=net_fwd,
        net_bwd=net_bwd,
        sig = args.sigma,
        eps=1e-4,
    )

    trainer.optimizer = {
        "forward": torch.optim.Adam(trainer.net_dict["forward"].parameters(), lr=1e-4),
        "backward": torch.optim.Adam(trainer.net_dict["backward"].parameters(), lr=1e-4),
    }

    trainer.train()

if __name__ == "__main__":
    main()
