# sde/bridge_sampler.py

"""This file gives us all the function to simulate SDE, for diffusion or for the brownian bridge"""

import torch
from torch import Tensor
from typing import Tuple
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def get_brownian_bridge(
    args,
    x_pairs: torch.Tensor,
    t_pairs: torch.Tensor,
    direction: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulates a Brownian bridge between t1 and t2 with z0 at t1 and z1 at t2.

    Args:
        args: object with attributes args.sigma, args.eps, and args.accelerator.device
        x_pairs: Tensor of shape [batch, 2, dim] containing (z0, z1)
        t_pairs: Tensor of shape [2] containing (t1, t2)
        direction: "forward" or "backward"

    Returns:
        z_t: sample from the Brownian bridge at time t
        t: sampled time tensor [batch, 1]
        target: estimated score (∇ log p(x_t))
    """

    device = args.accelerator.device
    z0, z1 = x_pairs[:, 0], x_pairs[:, 1]  # values at t1 and t2
    t1, t2 = t_pairs[:, 0], t_pairs[:, 1]  # time bounds reshaped to [batch, 1]

    t1 = t1.unsqueeze(1)  # [num_samples] -> [num_samples, 1]
    t2 = t2.unsqueeze(1)

    # Sample t uniformly in (t1 + eps, t2 - eps) to avoid sqrt(0) issues
    t = torch.rand((z0.shape[0], 1), device=device)

    t = t1 + (t2 - t1) * ((1 - 2 * args.eps) * t + args.eps)

    # Normalize time to [0, 1] as s = (t - t1) / (t2 - t1)
    s = (t - t1) / (t2 - t1)

    # Compute the mean of the Brownian bridge at time t
    z_t = (1 - s) * z0 + s * z1

    # Sample standard Gaussian noise
    z = torch.randn_like(z_t, device=device)

    # Add the stochastic part of the Brownian bridge
    z_t = z_t + args.sigma * torch.sqrt(s * (1 - s)) * z

    # Estimate the score depending on direction
    match direction:
        case "forward":
            target = (z1 - z0) - args.sigma * torch.sqrt(s / (1 - s)) * z
        case "backward":
            target = -(z1 - z0) - args.sigma * torch.sqrt((1 - s) / s) * z
        case _:
            raise ValueError("Direction must be 'forward' or 'backward'.")

    return z_t, t, target


@torch.no_grad()
def sample_sde(
    zstart: torch.Tensor,
    t_pairs: Tensor,
    net_dict,
    direction_tosample: str,
    N: int = 1000,
    sig: float = 1.0,
    device: str = "cuda",
):
    assert direction_tosample in ["forward", "backward"]

    t_min = t_pairs[:, 0]
    t_max = t_pairs[:, 1]

    # vectorisation of linspace method
    steps = torch.linspace(0, 1, N, device=device)  # shape [N]

    # expend t_min and t_max to create the linspace

    t_min_exp = t_min.unsqueeze(1)  # shape [6000, 1]
    t_max_exp = t_max.unsqueeze(1)  # shape [6000, 1]

    # Interpolation linéaire entre t_min et t_max
    ts = t_min_exp + (t_max_exp - t_min_exp) * steps  # shape [6000, N]

    if direction_tosample == "backward":
        ts = ts.flip(dims=[1])

    dt = torch.abs(t_max - t_min) / N
    dt = dt.view(-1, 1)  # shape: [6000, 1]

    z = zstart.detach().clone()
    score = net_dict[direction_tosample].eval()
    batchsize = z.size(0)  # TODO verif the dim if 0 is still the batchsize
    traj = [z.detach().clone()]


    for i in range(N-1): #TODO we change N in N-1 verif if its ok
        t = ts[:, i].unsqueeze(1)  # shape [batchsize, 1], chaque sample a son propre t
        pred = score(z, t)  # assume score accepte [batchsize, D] et [batchsize, 1]
        z = z + pred * dt + sig * torch.randn_like(z) * torch.sqrt(dt)
        traj.append(z.detach().clone())

    return traj, ts


@torch.no_grad()
def inference_sample_sde(
    zstart: torch.Tensor,
    t_pairs,
    net_dict,
    direction_tosample: str,
    N: int = 1000,
    sig: float = 1.0,
    device: str = "cuda",
):
    assert direction_tosample in ["forward", "backward"]

    t_min, t_max = t_pairs

    ts = np.linspace(t_min, t_max, N)

    if direction_tosample == "backward":
        ts = ts[::-1]

    dt = abs(t_max - t_min) / N

    z = zstart.detach().clone()

    score = net_dict[direction_tosample].eval()

    batchsize = z.size(0)

    traj = [z.detach().clone()]

    t_list = [ts[0]]  # TODO verif if this is okay, exploding variance at the end

    for i in range(N-1): #TODO cf function train for sample
        t = torch.full((batchsize, 1), float(ts[i]), device=device)

        t_list.append(ts[i])

        pred = score(z, t)

        z = z + pred * dt + sig * torch.randn_like(z) * np.sqrt(dt)

        traj.append(z.detach().clone())

    return traj, t_list
