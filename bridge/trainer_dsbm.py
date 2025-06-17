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
from utils.visualization import draw_trajectory_video
#from bridge.nbridges import N_BRIDGES

from utils.metric import get_classic_metrics



""" In this file we implement the main class for the training of 1 bridges, it represent the inner and outer
    iteration of Algorithm 1 of Shi &  Al 2023
"""


class IMF_DSBM:
    def __init__(
        self,
        x_pairs,
        x_pairs_test,
        T_min:int,
        T_max:int,
        args,
        num_simulation_steps: int,
        net_fwd: nn.Module,
        net_bwd: nn.Module,
        sig,
        eps
    ):
        
        # TODO comment each parameters of this function 

        super().__init__()

        self.x_pairs = x_pairs
        self.x_pairs_test = x_pairs_test

        self.T_min = T_min
        self.T_max = T_max
        self.args = args
        self.net_fwd = net_fwd
        self.net_bwd = net_bwd

        self.net_dict = {"forward": net_fwd, "backward": net_bwd}

        self.N = num_simulation_steps
        self.sig = sig
        self.eps = eps
        self.accelerator = args.accelerator
        self.device = args.accelerator.device




        self.net_dict["forward"].to(args.accelerator.device)
        self.net_dict["backward"].to(args.accelerator.device)


    def train(self): 
        


        self.losses_forward = []
        self.losses_backward = []

        for outer_iter_idx in range(self.args.nb_outer_iterations):

            loss_fwd = self.train_one_direction(direction= "forward", x_pairs = self.x_pairs, outer_iter_idx= outer_iter_idx)
            loss_bwd = self.train_one_direction(direction = "backward", x_pairs =  self.x_pairs, outer_iter_idx= outer_iter_idx)

            self.losses_forward.extend(loss_fwd)
            self.losses_backward.extend(loss_bwd)
        
        return self.net_dict
            
    def train_one_direction(self, x_pairs, direction: str, outer_iter_idx: int):

        loss_curve = []

        # 0. generate initial and final points
        dl = iter(DataLoader(
            TensorDataset(
                *self.generate_dataloaders(
                        args = self.args,
                        x_pairs= x_pairs,
                        direction_to_train= direction,
                        outer_iter_idx= outer_iter_idx,
                        first_coupling=self.args.first_coupling)),
                    
                batch_size= self.args.batch_size,
                shuffle = True,
                pin_memory= False,
                drop_last = True)
                )

        pbar = tqdm(range(self.args.nb_inner_opt_steps), desc=f"{direction} | Outer {outer_iter_idx}")

        for inner_opt_step in pbar:
            

            # 1. get batch of marginal points
            try :

                z0, z1 = next(dl) 
            
            except StopIteration:
                    

                    dl = iter(DataLoader(TensorDataset(*self.generate_dataloaders(
                        args = self.args,
                   
                        x_pairs= x_pairs,
                        direction_to_train= direction,
                        outer_iter_idx= outer_iter_idx,
                        first_coupling=self.args.first_coupling)), batch_size= self.args.batch_size, shuffle = True, pin_memory= False, drop_last = True)
                    )
            z_pairs = torch.stack([z0, z1], dim = 1).to(self.device)
            

            
            # 2. get Brownian bridge
            x_bridge_t, t, target = sampler.get_brownian_bridge(self.args, z_pairs, direction) 
            
            # 3. get net prediction

            pred = self.net_dict[direction](x_bridge_t, t)
            
            # 4. compute loss, backward, and optim step  #TODO deplace this in an abstract fonction which tract the train totally
            loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
            loss = loss.mean()
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.net_dict[direction].parameters(), self.args.grad_clip)
            self.optimizer[direction].step()
            self.optimizer[direction].zero_grad()
            
            pbar.set_postfix(loss=loss.item())
            loss_curve.append(loss.item())


        if outer_iter_idx % self.args.vis_every == 0:
            draw_trajectory_video(
                model=self,
                args=self.args,
                z0=x_pairs[:, 0], 
                z1=x_pairs[:, 1], 
                direction=direction,
                outer_iter_idx=outer_iter_idx,
                num_samples=1000,
                num_steps=self.args.num_simulation_steps,
                sigma=self.args.sigma
            )




            


        return loss_curve
    
    @torch.inference_mode()       
    def generate_dataloaders(self, args, x_pairs, direction_to_train: str, outer_iter_idx: int, first_coupling = None):
       
    
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
                net_dict = self.net_dict,
                direction_tosample = previous_direction,
                N = self.args.num_simulation_steps,
                sig = self.args.sigma,
                device = self.accelerator.device
            )[0][-1]

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
    


    @torch.inference_mode()
    def evaluate_moments_and_plot(self, direction: str, outer_iter_idx: int):
        if direction == "forward":
            ds_start = self.ds_t0_test
            ds_target = self.ds_t1_test
        else:
            ds_start = self.ds_t1_test
            ds_target = self.ds_t0_test

        target_points = []
        for batch in ds_target:
            x = batch[0].to(self.accelerator.device)
            target_points.append(x)
        target_tensor = torch.cat(target_points, dim=0)
        target_mean, target_std = get_classic_metrics(target_tensor.unsqueeze(0))  # [1, 2]

        start_points = []
        for batch in ds_start:
            x = batch[0].to(self.accelerator.device)
            start_points.append(x)
        start_tensor = torch.cat(start_points, dim=0)

        generated_tensor, _ = sampler.sample_sde(
            data=start_tensor,
            net_dict=self.net_dict,
            direction=direction,
            N=self.args.num_simulation_steps,
            sig=self.args.sigma,
            device=self.accelerator.device
        )

        gen_mean, gen_std = get_classic_metrics(generated_tensor.unsqueeze(0))  # [1, 2]


        if not hasattr(self, "mean_history"):
            self.mean_history = {"forward": [], "backward": []}
            self.std_history = {"forward": [], "backward": []}

        self.mean_history[direction].append(gen_mean.squeeze(0).cpu())
        self.std_history[direction].append(gen_std.squeeze(0).cpu())
        print(f"target_mean {target_mean}")
        print(f"target_std {target_std}")

        plot_moment(
            self.args,
            mean_gen = self.mean_history[direction],
            std_gen = self.std_history[direction],
            mean=target_mean,
            std=target_std,
            direction=direction
        )




    
    def plot_losses(self):
        """Plot loss curve for forward and backward"""


        plt.figure(figsize=(10, 5))
        plt.plot(self.losses_forward, label="Forward Loss")
        plt.plot(self.losses_backward, label="Backward Loss")
        plt.xlabel("Inner Step")
        plt.ylabel("Loss (MSE)")
        plt.title("Courbes de perte - Forward vs Backward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"loss.png") # TODO create a directory system for experiment 
        plt.close()
        

    @torch.inference_mode()
    def simulate_transport_trajectories(
        self,
        ds_start,
        ds_end,
        num_samples:int,
        num_step:int
    ):

        raise NotImplementedError

