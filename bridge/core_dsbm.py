#bridge/trainer_dsbm.py

import torch
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt


import bridge.sde.bridge_sampler as sampler
from utils.visualization import draw_plot, plot_moment


from utils.metric import get_classic_metrics



""" In this file we implement the main class for the training of 1 bridges, it represent the inner and outer
    iteration of Algorithm 1 of Shi &  Al 2023
"""


class IMF_DSBM():
    def __init__(
        self,
        args,
        num_simulation_steps: int,
        net_fwd: nn.Module,
        net_bwd: nn.Module,
        optimizer,
        sig,
        eps
    ):
        
        # TODO comment each parameters of this function 




        self.args = args
        self.accelerator = args.accelerator

        self.N = num_simulation_steps
        self.sig = sig
        self.eps = eps
        self.optimizer = optimizer
        self.device = args.accelerator.device


        self.net_fwd = net_fwd
        self.net_bwd = net_bwd

        self.net_dict = {}
        self.net_dict["forward"], self.net_dict["backward"], self.optimizer["forward"], self.optimizer["backward"] = \
        self.accelerator.prepare(net_fwd, net_bwd, optimizer["forward"], optimizer["backward"])






            
    def train_one_direction(self, x_pairs, t_pairs, num_bridges, direction: str, outer_iter_idx: int):

        loss_curve = []

        # 0. generate initial and final points
        dl = iter(self.accelerator.prepare(DataLoader(
            TensorDataset(
                *self.generate_dataloaders(
                        args = self.args,
                        x_pairs= x_pairs,
                        t_pairs = t_pairs,
                        direction_to_train= direction,
                        outer_iter_idx= outer_iter_idx,
                        first_coupling=self.args.first_coupling)),
                    
                batch_size= self.args.batch_size,
                shuffle = True,
                pin_memory= False,
                drop_last = True)
                ))

        pbar = tqdm(range(self.args.nb_inner_opt_steps), desc=f"{direction} | Outer {outer_iter_idx}")

        for inner_opt_step in pbar:
            

            # 1. get batch of marginal points
            try :

                z0, z1 = next(dl) 
            
            except StopIteration:
                    

                    dl = iter(self.accelerator.prepare(DataLoader(TensorDataset(*self.generate_dataloaders(
                        args = self.args,
                   
                        x_pairs= x_pairs,
                        t_pairs = t_pairs,
                        direction_to_train= direction,
                        outer_iter_idx= outer_iter_idx,
                        first_coupling=self.args.first_coupling)), batch_size= self.args.batch_size, shuffle = True, pin_memory= False, drop_last = True)
                    ))
            z_pairs = torch.stack([z0, z1], dim = 1).to(self.device)
            

            
            # 2. get Brownian bridge
            x_bridge_t, t, target = sampler.get_brownian_bridge(self.args, z_pairs, t_pairs, direction) 
            
            # 3. get net prediction

            pred = self.net_dict[direction](x_bridge_t, t)
            
            # 4. compute loss, backward, and optim step  
            loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
            loss = loss.mean()

            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.net_dict[direction].parameters(), self.args.grad_clip)
            self.optimizer[direction].step()
            self.optimizer[direction].zero_grad()
            
            pbar.set_postfix(loss=loss.item())
            loss_curve.append(loss.item())
            



        if outer_iter_idx % self.args.vis_every == 0:

            draw_plot(
                model=self,
                args=self.args,
                num_bridges=num_bridges,
                z0=x_pairs[:,0], 
                z1=x_pairs[:,1],
                t_pairs = t_pairs,
                direction=direction,
                outer_iter_idx=outer_iter_idx,
                num_samples=1000,
                num_steps=self.args.num_simulation_steps,
                sigma=self.args.sigma
            )

        return loss_curve
    
    @torch.inference_mode()       
    def generate_dataloaders(self, args, x_pairs, t_pairs, direction_to_train: str, outer_iter_idx: int, first_coupling = None):
       
    
        if outer_iter_idx == 0 and direction_to_train == "forward":

            if first_coupling == 'ref':

                zstart = x_pairs[:, 0]
                zend = zstart + torch.randn_like(zstart)* args.sigma



            
            elif first_coupling == 'ind' :

                zstart = x_pairs[:, 0]
                zend = x_pairs[:, 1].clone()
                zend = zend[torch.randperm(len(zend))]
            
            else:
                raise NotImplementedError

            z0, z1 = zstart, zend

        
        # if not first it    
        else:


            match direction_to_train:

                case "forward": # previous = backward
                    
                    zstart = x_pairs[:, 1] # begin with t1
                    previous_direction = "backward"

                case "backward": # previous = forward

                    zstart = x_pairs[:, 0] # begin with t0 
                    previous_direction = "forward"

                case _:
                    raise ValueError(f"Unknown direction: {direction_to_train}")
                
            zend = sampler.sample_sde(

                zstart = zstart,
                t_pairs= t_pairs,
                net_dict = self.net_dict,
                direction_tosample = previous_direction,
                N = self.args.num_simulation_steps,
                sig = self.args.sigma,
                device = self.accelerator.device
            )[-1]

            match direction_to_train:

                case "forward": # previous = backward
                    
                    z0, z1 = zend, zstart

                case "backward": # previous = forward

                    z0, z1 = zstart, zend

                case _:
                    raise ValueError(f"Unknown direction: {direction_to_train}")
        

        return z0, z1
    
    def compute_loss(
            self,
            pred : Tensor,
            target: Tensor

    ) -> Tensor:
        
        return F.mse_loss(pred,target)
    


   