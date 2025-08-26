# bridge/trainer_dsbm.py

import torch
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt


import bridge.sde.bridge_sampler as sampler


""" In this file we implement the main class for the training of 1 bridges, it represent the inner and outer
    iteration of Algorithm 1 of Shi &  Al 2023
"""


class IMF_DSBM:
    def __init__(
        self,
        args,
        min_time: float,
        max_time: float,
        num_simulation_steps: int,
        net_fwd: nn.Module,
        net_bwd: nn.Module,
        net_fwd_ema,
        net_bwd_ema,
        optimizer,
        sig,
        eps,
    ):
        # TODO comment each parameters of this function
        self.first_pass = True
        self.args = args
        self.accelerator = args.accelerator

        self.N = num_simulation_steps
        self.min_time = min_time
        self.max_time = max_time
        self.sig = sig
        self.eps = eps
        self.optimizer = optimizer
        self.device = args.accelerator.device

        self.net_fwd = net_fwd
        self.net_bwd = net_bwd

        self.net_fwd_ema = net_fwd_ema
        self.net_bwd_ema = net_bwd_ema

        self.net_dict = {}
        (
            self.net_dict["forward"],
            self.net_dict["backward"],
            self.optimizer["forward"],
            self.optimizer["backward"],
        ) = self.accelerator.prepare(
            net_fwd, net_bwd, optimizer["forward"], optimizer["backward"]
        )

        self.ema_dict = {
            "forward": self.net_fwd_ema,
            "backward": self.net_bwd_ema,
        }

        self.ema_dict["forward"].to(self.device)
        self.ema_dict["backward"].to(self.device)

    def train_one_direction(
        self, x_pairs, t_pairs: Tensor, direction: str, outer_iter_idx: int
    ):
        # x_pairs shape (num_distrib, num_samples, time, dim)

        loss_curve = []
        grad_curve = []
        # 0. generate initial and final points
        dataset = TensorDataset(
            *self.generate_dataloaders(
                args=self.args,
                x_pairs=x_pairs,
                t_pairs=t_pairs,
                direction_to_train=direction,
                outer_iter_idx=outer_iter_idx,
                first_coupling=self.args.first_coupling,
            )
        )
        dl = iter(
            self.accelerator.prepare(
                DataLoader(
                    dataset,
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    pin_memory=False,
                    drop_last=True,
                )
            )
        )

        nb_inner_opt_steps = self.args.nb_inner_opt_steps

        if self.args.warmup and outer_iter_idx == 0 and self.first_pass:
            print(f"WARMUP STEP {direction}")

            nb_inner_opt_steps = self.args.warmup_nb_inner_opt_steps

        # at this step we have dataloader with initial point generate and real end point
        pbar = tqdm(
            range(nb_inner_opt_steps),
            desc=f"{direction} | Outer {outer_iter_idx}",
        )

        for inner_opt_step in pbar:
            # 1. get batch of marginal points
            try:
                z0, z1, t_tensor = next(dl)

            except StopIteration:
                del dl , 

                self.clear()
                dataset = TensorDataset(
                *self.generate_dataloaders(
                    args=self.args,
                    x_pairs=x_pairs,
                    t_pairs=t_pairs,
                    direction_to_train=direction,
                    outer_iter_idx=outer_iter_idx,
                    first_coupling=self.args.first_coupling,
            ))
                dl = iter(
                    self.accelerator.prepare(
                        DataLoader(
                            dataset,
                            batch_size=self.args.batch_size,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True,
                        )
                    )
                )
                z0, z1, t_tensor = next(dl)

            z_pairs = torch.stack([z0, z1], dim=1).to(self.device)

            # 2. get Brownian bridge
            x_bridge_t, t, target, sigma = sampler.get_brownian_bridge(
                self.args, z_pairs, t_tensor, direction
            )

            if self.args.loss_scale:
                if direction == "forward":
                    loss_scale = sigma * torch.sqrt(t)

                elif direction == "backward":
                    loss_scale = sigma * torch.sqrt(self.max_time - t)

            else:
                loss_scale = 1

            # 3. get net prediction

            pred = self.net_dict[direction](x_bridge_t, t)

            # 4. compute loss, backward, and optim step
            loss = F.mse_loss(loss_scale * pred, loss_scale * target)

            self.optimizer[direction].zero_grad(set_to_none=True)
            self.accelerator.backward(loss)

            # Compute total gradient norm (L2 norm)
            total_norm = 0.0
            for p in self.net_dict[direction].parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5  # Final L2 norm of all gradients

            self.accelerator.clip_grad_norm_(
                self.net_dict[direction].parameters(), self.args.grad_clip
            )
            self.optimizer[direction].step()
            if (
                outer_iter_idx == 0
                and inner_opt_step == nb_inner_opt_steps - 1
            ):
                self.ema_dict[direction].ema_model.load_state_dict(
                    self.net_dict[direction].state_dict()
                )

            self.ema_dict[direction].update(self.net_dict[direction])

            pbar.set_postfix(loss=loss.item())
            loss_curve.append(loss.item())
            grad_curve.append(total_norm)

        self.first_pass = False
        self.clear()
        return (
            loss_curve,
            grad_curve,
            {"forward": self.net_fwd, "backward": self.net_bwd},
            {"forward": self.net_fwd_ema, "backward": self.net_bwd_ema},
        )
    
    def clear(self):
        self.accelerator.free_memory()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate_dataloaders(
        self,
        args,
        x_pairs,
        t_pairs: Tensor,
        direction_to_train: str,
        outer_iter_idx: int,
        first_coupling=None,
    ):
        if outer_iter_idx <= self.args.warmup_epoch and self.first_pass:
            if direction_to_train == "forward" :
                if first_coupling == "ref":
                    zstart = x_pairs[:, 0]
                    zend = (
                        zstart + torch.randn_like(zstart) * args.sigma
                    )  # TODO see if the perturbation need to be of the same std of the stepsize

                elif first_coupling == "ind":

                    zstart = x_pairs[:, 0]
                    zend = x_pairs[:, 1].clone()
                    for t_pair in torch.unique(t_pairs, dim=0):

                        indices = (t_pairs == t_pair).all(dim=1).nonzero(as_tuple=True)[0].tolist()
                        zend_t_pair = zend[indices]

                        permutation = torch.randperm(len(indices))
                        zend_permuted = zend_t_pair[permutation]

                        zend[indices] = zend_permuted
                    
                    final_permutation = torch.randperm(zstart.shape[0])
                    zstart = zstart[final_permutation]
                    zend = zend[final_permutation]
                    t_pairs = t_pairs[final_permutation]

                else:
                    raise NotImplementedError

                z0, z1 = zstart, zend

            elif direction_to_train == "backward":
                if first_coupling == "ref":
                    zstart = x_pairs[:, 1]
                    zend = (
                        zstart + torch.randn_like(zstart) * args.sigma
                    )  # TODO see if the perturbation need to be of the same std of the stepsize

                elif first_coupling == "ind":
                    zstart = x_pairs[:, 1]
                    zend = x_pairs[:, 0].clone()
                    for t_pair in torch.unique(t_pairs, dim=0):

                        indices = (t_pairs == t_pair).all(dim=1).nonzero(as_tuple=True)[0].tolist()
                        zend_t_pair = zend[indices]

                        permutation = torch.randperm(len(indices))
                        zend_permuted = zend_t_pair[permutation]

                        zend[indices] = zend_permuted
                    
                    final_permutation = torch.randperm(zstart.shape[0])
                    zstart = zstart[final_permutation]
                    zend = zend[final_permutation]
                    t_pairs = t_pairs[final_permutation]

                else:
                    raise NotImplementedError

                z0, z1 = zend, zstart

        # if not first it
        else:
            match direction_to_train:
                case "forward":  # previous = backward
                    zstart = x_pairs[:, 1]  # begin with t1
                    previous_direction = "backward"

                case "backward":  # previous = forward
                    zstart = x_pairs[:, 0]  # begin with t0
                    previous_direction = "forward"

                case _:
                    raise ValueError(f"Unknown direction: {direction_to_train}")



            zend = sampler.sample_sde(  # TODO see sample SDE
                args,
                zstart=zstart,
                t_pairs=t_pairs,
                full_traj_tmin=self.min_time,
                full_traj_tmax=self.max_time,
                net_dict=self.ema_dict,
                direction_tosample=previous_direction,
                N=self.args.num_simulation_steps,
                device=self.accelerator.device,
            )

            match direction_to_train:
                case "forward":  # previous = backward
                    z0, z1 = zend, zstart

                case "backward":  # previous = forward
                    z0, z1 = zstart, zend

                case _:
                    raise ValueError(f"Unknown direction: {direction_to_train}")

        return z0, z1, t_pairs
