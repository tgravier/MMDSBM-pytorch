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

    t_pairs_bridges = args.t_pairs_bridges
    # Determine sigma: if it's a float, use as is; if it's a sequence, select per sample
    if args.sigma_mode == "mono":
        sigma = args.sigma
    elif args.sigma_mode == "multi":
        sigma = torch.zeros_like(t1)
        # Assume args.sigma is a tensor or list with shape [batch] or broadcastable
        if direction == "forward":
            for i, (start, end) in enumerate(t_pairs_bridges):
                mask = (t >= start) & (t < end)
                sigma[mask] = args.sigma[i + 1]

        elif direction == "backward":
            for i, (start, end) in enumerate(t_pairs_bridges):
                mask = (t > start) & (t <= end)
                sigma[mask] = args.sigma[i]

    elif args.sigma_mode == "multi_dim":
        sigma = torch.zeros_like(z0)

        if direction == "forward":
            for i, (start, end) in enumerate(t_pairs_bridges):
                mask = (t >= start) & (t < end)
                indices = torch.where(mask.squeeze())[0]

                if args.sigma_linspace == "final":
                    sigma[indices, :] = args.sigma_tensor_list[i + 1, :].unsqueeze(0)
                elif args.sigma_linspace == "linear":
                    sigma[indices, :] = (1 - s[indices]) * args.sigma_tensor_list[
                        i, :
                    ].unsqueeze(0) + s[indices] * args.sigma_tensor_list[i + 1, :].unsqueeze(0)

        elif direction == "backward":
            for i, (start, end) in enumerate(t_pairs_bridges):
                mask = (t >= start) & (t < end)
                indices = torch.where(mask.squeeze())[0]

                if args.sigma_linspace == "final":
                    sigma[indices, :] = args.sigma_tensor_list[i, :].unsqueeze(0)
                elif args.sigma_linspace == "linear":
                    sigma[indices, :] = (1 - s[indices]) * args.sigma_tensor_list[
                        i, :
                    ].unsqueeze(0) + s[indices] * args.sigma_tensor_list[i + 1, :].unsqueeze(0)

    # Compute the mean of the Brownian bridge at time t
    z_t = (1 - s) * z0 + s * z1

    # Sample standard Gaussian noise
    z = torch.randn_like(z_t, device=device)
    sigma = args.coeff_sigma * sigma
    # Add the stochastic part of the Brownian bridge
    z_t = z_t + sigma * torch.sqrt(s * (1 - s)) * z

    # Estimate the score depending on direction
    match direction:
        case "forward":
            target = (z1 - z0) - sigma * torch.sqrt(s / (1 - s)) * z
        case "backward":
            target = -(z1 - z0) - sigma * torch.sqrt((1 - s) / s) * z
        case _:
            raise ValueError("Direction must be 'forward' or 'backward'.")

    return z_t, t, target, sigma


@torch.no_grad()
def sample_sde(
    args,
    zstart: torch.Tensor,
    t_pairs: Tensor,
    net_dict,
    direction_tosample: str,
    N: int,
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
        if direction_tosample == "forward":
            this_time_delta_ts = this_time_delta_ts[
                :, :-1
            ]  
        elif direction_tosample == "backward":
            this_time_delta_ts = this_time_delta_ts[:, 1:]

        # Pad this_time_delta_ts with NaNs to have shape [num_samples, max_nb_steps]
        pad_size = max_nb_steps - this_time_delta_ts.shape[1]
        if pad_size > 0:
            nan_pad = torch.full(
                (this_time_delta_ts.shape[0], pad_size), float("nan"), device=device
            )
            this_time_delta_ts = torch.cat([this_time_delta_ts, nan_pad], dim=1)

        ts[this_time_delta_indices] = this_time_delta_ts

    dt = 1.0 / (N * tensor_of_proportions_of_total)
    dt = dt.unsqueeze(1)

    if direction_tosample == "backward":
        ts = ts.flip(dims=[1])
        dt = dt.flip(dims=[1])

    t_pairs_bridges = args.t_pairs_bridges

    # Determine sigma: if it's a float, use as is; if it's a sequence, select per sample
    if args.sigma_mode == "mono":
        sigma = args.sigma
        sigma = torch.full((len(ts),), args.sigma, device=device)
    elif args.sigma_mode == "multi":
        sigma = torch.zeros_like(ts)
        # Assume args.sigma is a tensor or list with shape [batch] or broadcastable
        if direction_tosample == "forward":
            for i, (start, end) in enumerate(t_pairs_bridges):
                mask = (ts >= start) & (ts < end)
                sigma[mask] = args.sigma[i + 1]

        elif direction_tosample == "backward":
            for i, (start, end) in enumerate(t_pairs_bridges):
                mask = (ts > start) & (ts <= end)
                sigma[mask] = args.sigma[i]

    elif args.sigma_mode == "multi_dim":
        dim = zstart.shape[1]
        nb_simulation_step = ts.shape[0]
        sigma = torch.zeros((nb_simulation_step, dim), device=args.accelerator.device)

        if direction_tosample == "forward":
            for i, (start, end) in enumerate(t_pairs_bridges):
                mask = (ts >= start) & (ts < end)
                indices = torch.where(mask.squeeze())[0]


                if args.sigma_linspace == "final":
                     
                    sigma[indices, :] = args.sigma_tensor_list[i+1, :].unsqueeze(0)

                elif args.sigma_linspace == "linear":
                     
                     sigma = torch.zeros((ts.shape[0], ts.shape[1], dim), device=args.accelerator.device)

                     s = (ts - start) / (end - start)   # shape (nb_samples, max_nb_steps)
                     sigma_i = args.sigma_tensor_list[i, :].view(1, 1, -1)         # (1, 1, dim)
                     sigma_ip1 = args.sigma_tensor_list[i + 1, :].view(1, 1, -1)   # (1, 1, dim)
                     s_exp = s.unsqueeze(-1)  # (nb_samples, max_nb_steps, 1)
                     sigma_interp = (1 - s_exp) * sigma_i + s_exp * sigma_ip1      # (nb_samples, max_nb_steps, dim)
                     sigma[mask] = sigma_interp[mask]


        elif direction_tosample == "backward":
            for i, (start, end) in enumerate(t_pairs_bridges):
                mask = (ts > start) & (ts <= end)
                indices = torch.where(mask.squeeze())[0]

                if args.sigma_linspace == "final":
                     
                    sigma[indices, :] = args.sigma_tensor_list[i, :].unsqueeze(0)

                elif args.sigma_linspace == "linear":
                     sigma = torch.zeros((ts.shape[0], ts.shape[1], dim), device=args.accelerator.device)
                     s = (ts - start) / (end - start)   # shape (nb_samples, max_nb_steps)
                     sigma_i = args.sigma_tensor_list[i, :].view(1, 1, -1)         # (1, 1, dim)
                     sigma_ip1 = args.sigma_tensor_list[i + 1, :].view(1, 1, -1)   # (1, 1, dim)
                     s_exp = s.unsqueeze(-1)  # (nb_samples, max_nb_steps, 1)
                     sigma_interp = (1 - s_exp) * sigma_i + s_exp * sigma_ip1      # (nb_samples, max_nb_steps, dim)
                     sigma[mask] = sigma_interp[mask]


    z = zstart.detach().clone()
    score = net_dict[direction_tosample].eval()

    sigma = args.coeff_sigma * sigma

    for i in range(1,max_nb_steps):  # TODO: clean the [mask]s # Starting from one to avoid the first nan which add just noise
        mask = ~torch.isnan(ts[:, i])  # mask to filter out NaNs
        t = ts[mask, i].unsqueeze(
            1
        )  # shape [batchsize, 1], chaque sample a son propre t
        pred = score(
            z[mask], t
        )  # assume score accepte [batchsize, D] et [batchsize, 1]

        if args.sigma_mode == "linear":

            z[mask] = (
                z[mask]
                + pred * dt[mask]
                + sigma[mask, i,:]
                * torch.randn_like(z[mask])
                * torch.sqrt(dt[mask])
        
        )
            
        else:


            z[mask] = (
                z[mask]
                + pred * dt[mask]
                + sigma[mask,:]
                * torch.randn_like(z[mask])
                * torch.sqrt(dt[mask])
        
        )


    return z


@torch.no_grad()
def inference_sample_sde(
    args,
    zstart: torch.Tensor,
    t_pairs: torch.Tensor,
    device: str,
    net_dict,
    direction_tosample: str,
    N: int = 1000,
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
        ts = torch.arange(full_traj_tmax, full_traj_tmin, step=-dt_step, device=device)
        dt = dt.flip(dims=[0])

    t_pairs_bridges = args.t_pairs_bridges
    # Determine sigma: if it's a float, use as is; if it's a sequence, select per sample
    if args.sigma_mode == "mono":
        sigma = args.sigma
        sigma = torch.full((len(ts),), args.sigma, device=device)
    elif args.sigma_mode == "multi":
        sigma = torch.zeros_like(ts)
        # Assume args.sigma is a tensor or list with shape [batch] or broadcastable
        if direction_tosample == "forward":
            for i, (start, end) in enumerate(t_pairs_bridges):
                mask = (ts >= start) & (ts < end)

                sigma[mask] = args.sigma[i + 1]

        elif direction_tosample == "backward":
            for i, (start, end) in enumerate(t_pairs_bridges):
                mask = (ts > start) & (ts <= end)
                sigma[mask] = args.sigma[i]

    elif args.sigma_mode == "multi_dim":
        dim = zstart.shape[1]
        nb_simulation_step = ts.shape[0]
        sigma = torch.zeros((nb_simulation_step, dim), device=args.accelerator.device)

        if direction_tosample == "forward":
            for i, (start, end) in enumerate(t_pairs_bridges):
                mask = (ts >= start) & (ts < end)
                indices = torch.where(mask.squeeze())[0]

                if args.sigma_linspace == "final":
                    sigma[indices, :] = args.sigma_tensor_list[i + 1, :].unsqueeze(0)

                elif args.sigma_linspace == "linear":
                    s = (ts[indices] - start) / (end - start)

                    sigma[indices, :] = (1 - s).unsqueeze(1) * args.sigma_tensor_list[
                        i, :
                    ].unsqueeze(0) + s.unsqueeze(1) * args.sigma_tensor_list[i + 1, :].unsqueeze(0)

        elif direction_tosample == "backward":
            for i, (start, end) in enumerate(t_pairs_bridges):
                mask = (ts > start) & (ts <= end)
                indices = torch.where(mask.squeeze())[0]

                if args.sigma_linspace == "final":
                    sigma[indices, :] = args.sigma_tensor_list[i, :].unsqueeze(0)

                elif args.sigma_linspace == "linear":
                    s = (ts[indices] - start) / (end - start)

                    sigma[indices, :] = (1 - s).unsqueeze(1) * args.sigma_tensor_list[
                        i, :
                    ].unsqueeze(0) + s.unsqueeze(1) * args.sigma_tensor_list[i + 1, :].unsqueeze(0)

    sigma = args.coeff_sigma * sigma
    score = net_dict[direction_tosample].eval()

    z = zstart.detach().clone()

    traj = [z.detach().clone()]
    t_list = [ts[0]]

    batchsize = z.shape[0]

    for i in range(len(ts)):
        t = torch.full((batchsize, 1), float(ts[i]), device=device)

        t_list.append(ts[i] + dt_step * sign)

        pred = score(z, t)

        z = z + pred * dt[i] + sigma[i] * torch.randn_like(z) * torch.sqrt(dt[i])

        traj.append(z.detach().clone())

    return traj, t_list
