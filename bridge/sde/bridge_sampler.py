# sde/bridge_sampler.py

"""This file gives us all the function to simulate SDE, for diffusion or for the brownian bridge"""

# TODO verifiate each line of this function to see if we have the best sampler for train & inference

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
    N: int,
    sig: float,
    device: str,
    full_traj_tmax: float,
    full_traj_tmin: float,
):
    assert direction_tosample in ["forward", "backward"]

    t_min = t_pairs[:, 0]
    t_max = t_pairs[:, 1]

    time_deltas = t_max - t_min
    
    tensor_of_proportions_of_total = time_deltas / (full_traj_tmax - full_traj_tmin)

    # create the linspace for each time delta proportionally to the total time of the full trajectory
    proportions_of_total = {
        one_time_delta: one_time_delta / (full_traj_tmax - full_traj_tmin)
        for one_time_delta in torch.unique(time_deltas)
    }
    # print(f"DEBUG proportions_of_total: {proportions_of_total}")
    steps = {
        one_time_delta: torch.linspace(
            0,
            1,
            int(torch.round(N * proportions_of_total[one_time_delta]).item()),
            device=device,
        )
        for one_time_delta in proportions_of_total
    }
    max_nb_steps = max(len(s) for s in steps.values()) 

    # expend t_min and t_max to create the linspace
    t_min_exp = t_min.unsqueeze(1)  # shape [6000, 1]
    t_max_exp = t_max.unsqueeze(1)  # shape [6000, 1]

    # Interpolation linéaire entre t_min et t_max
    ts = torch.full(
        (len(t_min), max_nb_steps), float("nan"), dtype=torch.float, device=device
    )  # shape [6000, N]
    this_time_delta_indices_list = []

    for one_time_delta in steps:

        this_time_delta_indices = time_deltas == one_time_delta
        this_time_delta_t_min_exp = t_min_exp[this_time_delta_indices]
        this_time_delta_t_max_exp = t_max_exp[this_time_delta_indices]
        this_time_delta_ts = (
            this_time_delta_t_min_exp
            + (this_time_delta_t_max_exp - this_time_delta_t_min_exp)
            * steps[one_time_delta]
        )
        # if direction_tosample == "forward":
        #     this_time_delta_ts = this_time_delta_ts[:, :-1]

        # elif direction_tosample == "backward":
        #     this_time_delta_ts = this_time_delta_ts[:, 1:]
        


        # Pad this_time_delta_ts with NaNs to have shape [num_samples, max_nb_steps]
        pad_size = max_nb_steps - this_time_delta_ts.shape[1]
        if pad_size > 0:
            nan_pad = torch.full(
                (this_time_delta_ts.shape[0], pad_size), float("nan"), device=device
            )
            this_time_delta_ts = torch.cat([this_time_delta_ts, nan_pad], dim=1)

        ts[this_time_delta_indices] = this_time_delta_ts



    dt = 1.0/(N*tensor_of_proportions_of_total) 
    dt = dt.unsqueeze(1)

    if direction_tosample == "backward":
        ts = ts.flip(dims=[1])
        dt = dt.flip(dims=[1])

    z = zstart.detach().clone()
    score = net_dict[direction_tosample].eval()

    for i in range(max_nb_steps):  # TODO: clean the [mask]s
        mask = ~torch.isnan(ts[:, i])  # mask to filter out NaNs
        t = ts[mask, i].unsqueeze(
            1
        )  # shape [batchsize, 1], chaque sample a son propre t
        pred = score(
            z[mask], t
        )  # assume score accepte [batchsize, D] et [batchsize, 1]
        z[mask] = (
            z[mask]
            + pred * dt[mask]
            + sig * torch.randn_like(z[mask]) * torch.sqrt(dt[mask])
        )
    return z


@torch.no_grad()
def inference_sample_sde(
    zstart: torch.Tensor,
    t_pairs: torch.Tensor,
    device: str,
    net_dict,
    direction_tosample: str,
    N: int = 1000,
    sig: float = 1.0,
):
    assert direction_tosample in ["forward", "backward"]

    full_traj_tmax = t_pairs.cpu().max()
    full_traj_tmin = t_pairs.cpu().min()

    sign = 1.0

    dt_step = (full_traj_tmax - full_traj_tmin) / N

    ts = torch.arange(full_traj_tmin, full_traj_tmax, step=dt_step, device=device)

    # Compute dt for each interval in t_pairs
    dt = torch.zeros_like(ts, device=device)
    for i in range(t_pairs.shape[0]):
        t_start = t_pairs[i, 0]
        t_end = t_pairs[i, 1]
        mask = (ts >= t_start) & (ts < t_end)
        interval_dt = (full_traj_tmax - full_traj_tmin) / ((t_end - t_start) * N)
        dt[mask] = interval_dt

    if direction_tosample == "backward":
        sign = -1
        ts = torch.arange(full_traj_tmax, full_traj_tmin, step=-dt_step)
        dt = dt.flip(dims=[0])

    score = net_dict[direction_tosample].eval()

    z = zstart.detach().clone()

    traj = [z.detach().clone()]
    t_list = [ts[0]]

    batchsize = z.shape[0]

    for i in range(len(ts)):
        t = torch.full((batchsize, 1), float(ts[i]), device=device)

        t_list.append(ts[i] + dt_step * sign)

        pred = score(z, t)

        z = z + pred * dt[i] + sig * torch.randn_like(z) * torch.sqrt(dt[i])

        traj.append(z.detach().clone())

    return traj, t_list
