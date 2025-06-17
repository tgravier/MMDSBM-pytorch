import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from accelerate import Accelerator
from types import SimpleNamespace

from bridge.trainer_dsbm import IMF_DSBM
from bridge.models.networks import ScoreNetwork


def make_circle(n_samples, center, radius, thickness=0.05):
    angles = np.random.uniform(0, 2 * np.pi, n_samples)
    radii = np.random.normal(loc=radius, scale=thickness, size=n_samples)
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return np.stack([x, y], axis=1)


def sample_ring_circle_to_circle(n_samples):
    x0 = make_circle(n_samples, center=[2.0, 2.0], radius=1.0, thickness=0.05)
    x1 = make_circle(n_samples, center=[7.0, 7.0], radius=1.5, thickness=0.08)
    return torch.tensor(x0, dtype=torch.float32), torch.tensor(x1, dtype=torch.float32)


def main():
    args = SimpleNamespace(
        seed=42,
        experiment_dir="experiments",
        experiment_name="circle_circle_02",
        dim=2,
        first_coupling="ref",
        sigma=1.0,
        num_simulation_steps=20,
        nb_inner_opt_steps=10000,
        nb_outer_iterations=20,
        eps=1e-3,
        batch_size=128,
        grad_clip=1.0,
        vis_every=1,
        accelerator=Accelerator(),
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Using device: {args.accelerator.device}")

    # -------- Création des cercles pour visualisation --------
    circle1 = make_circle(
        n_samples=2000,
        center=[2.0, 2.0],
        radius=1.0,
        thickness=0.05,
    )

    circle2 = make_circle(
        n_samples=2000,
        center=[7.0, 7.0],
        radius=1.5,
        thickness=0.08,
    )


    # -------- Données pour entraînement --------
    x0, x1 = sample_ring_circle_to_circle(n_samples=5000)
    x_pairs = torch.stack([x0, x1], dim=1).to(args.accelerator.device)

    x0_test, x1_test = sample_ring_circle_to_circle(n_samples=2000)
    x_pairs_test = torch.stack([x0_test, x1_test], dim=1).to(args.accelerator.device)

    # -------- Score networks --------
    net_fwd = ScoreNetwork(
        input_dim=args.dim + 1,
        layers_widths=[128, 128, args.dim],
        activation_fn=nn.SiLU(),
    )
    net_bwd = ScoreNetwork(
        input_dim=args.dim + 1,
        layers_widths=[128, 128, args.dim],
        activation_fn=nn.SiLU(),
    )

    # -------- Entraîneur --------
    trainer = IMF_DSBM(
        x_pairs=x_pairs,
        x_pairs_test=x_pairs_test,
        T_min=0.0,
        T_max=1.0,
        args=args,
        num_simulation_steps=args.num_simulation_steps,
        net_fwd=net_fwd,
        net_bwd=net_bwd,
        sig=args.sigma,
        eps=1e-4,
    )

    trainer.optimizer = {
        "forward": torch.optim.Adam(trainer.net_dict["forward"].parameters(), lr=1e-4),
        "backward": torch.optim.Adam(
            trainer.net_dict["backward"].parameters(), lr=1e-4
        ),
    }

    # -------- Lancement de l'entraînement --------
    trainer.train()


if __name__ == "__main__":
    main()
