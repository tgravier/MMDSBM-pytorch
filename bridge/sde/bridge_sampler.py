#sde/bridge_sampler.py

""" This file gives us all the function to simulate SDE, for diffusion or for the brownian bridge"""

import torch
from torch import Tensor
from typing import Tuple
import numpy as np
from torch.utils.data import DataLoader, TensorDataset



def get_brownian_bridge(args,
                        x_pairs,
                        direction: str,
                        ) -> Tuple[Tensor, Tensor, Tensor]:



    device = args.accelerator.device
    z0, z1 = x_pairs[:, 0], x_pairs[:, 1] # direction t0 -> t1

    

    t = torch.rand((z1.shape[0], 1), device= device) * (1-2*args.eps) + args.eps # to avoid sqrt(neg) during brownian bridge
    


    z_t = t * z1 + (1. - t)* z0
    

    z = torch.randn_like(z_t, device=device) # Wiener noise with the same shape of x_t

    z_t = z_t + args.sigma * torch.sqrt(t*(1-t)) * z # brownian bridge


    match direction:

        case "forward":

            # z1 - z_t / (1-t)
            target = z1 - z0 #TODO please be careful t between 0,1 becasue we can have T_min & T_max for the marginals
            target = target - args.sigma * torch.sqrt(t/(1.-t)) * z

        case "backward":


            # z0 - z_t / t 
            target = - (z1 - z0) #TODO verif if its x_final - x_start or the other way
            target = target - args.sigma * torch.sqrt((1.-t)/t) * z


    return z_t, t, target #return : z_t : brownian bridge at time t, t uniform [0,1], target : score estimation




@torch.no_grad()
def sample_sde(zstart: torch.Tensor, net_dict, direction_tosample: str, N: int = 1000, sig: float = 1.0, device: str = "cuda"): #TODO put T_min and T_max 
    assert direction_tosample in ['forward', 'backward']


    dt = 1. / N
    ts = np.arange(N) / N
    if direction_tosample == "backward":
        ts = 1 - ts


    z = zstart.detach().clone()
    score = net_dict[direction_tosample].eval()
    batchsize = z.size(0)
    traj = [z.detach().clone()]

    

    for i in range(N):
        t = torch.full((batchsize, 1), ts[i], device=device)
        pred = score(z, t) 
        z = z.detach().clone() + pred * dt
        z = z + sig * torch.randn_like(z) * np.sqrt(dt)
        traj.append(z.detach().clone())

    

    return traj, ts



