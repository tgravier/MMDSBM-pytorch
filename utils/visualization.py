#utils/visualisation.py

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
from torch import Tensor
from scipy.stats import gaussian_kde

from bridge.sde.bridge_sampler import sample_sde
from utils.metric import get_classic_metrics

from typing import List



@torch.no_grad()
def get_n_samples_from_loader(dataloader: DataLoader, num_samples: int, batch_size: int, device="cpu") -> DataLoader:
    samples = []

    for batch in dataloader:
        x = batch[0].to(device)
        samples.append(x)
        if sum([s.shape[0] for s in samples]) >= num_samples:
            break

    x = torch.cat(samples, dim=0)[:num_samples]
    dataset = TensorDataset(x)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)





@torch.no_grad()
def draw_plot(
    model,
    args,
    z0: torch.Tensor,
    z1: torch.Tensor,
    direction: str,
    outer_iter_idx: int,
    num_samples: int = 1000,
    num_steps: int = 1000,
    sigma: float = 1.0,
):
    experiment_name_folder = os.path.join(args.experiment_dir, args.experiment_name)
    folder_fig = os.path.join(experiment_name_folder, "figs")
    os.makedirs(folder_fig, exist_ok=True)

    if z0.dim() == 3:
        z0 = z0.view(-1, z0.shape[-1])
    if z1.dim() == 3:
        z1 = z1.view(-1, z1.shape[-1])

    # sample
    idx = torch.randint(0, z0.shape[0], (num_samples,))
    z0_sampled = z0[idx].to(args.accelerator.device)
    z1_sampled = z1[idx].to(args.accelerator.device)

    # generate samples
    if direction == 'forward':
        generated_samples = sample_sde(
            zstart=z0_sampled,
            net_dict=model.net_dict,
            direction_tosample=direction,
            N=num_steps,
            sig=sigma,
            device=args.accelerator.device
        )[-1]
    elif direction == 'backward':
        generated_samples = sample_sde(
            zstart=z1_sampled,
            net_dict=model.net_dict,
            direction_tosample=direction,
            N=num_steps,
            sig=sigma,
            device=args.accelerator.device
        )[-1]

    # convert to numpy
    z0_np = z0_sampled.cpu().numpy()
    z1_np = z1_sampled.cpu().numpy()
    zgen_np = generated_samples.cpu().numpy()

    # plot setup
    plt.figure(figsize=(8, 8))

    # scatter plots
    plt.scatter(z0_np[:, 0], z0_np[:, 1], s=4, alpha=0.8, label="Initial", color='blue')   # smaller
    plt.scatter(z1_np[:, 0], z1_np[:, 1], s=4, alpha=0.8, label="Target", color='red')    # smaller
    plt.scatter(zgen_np[:, 0], zgen_np[:, 1], s=10, alpha=0.6, label="Generated", color='green')  # larger

    # KDE contour helper
    def draw_contour(data, color):
        kde = gaussian_kde(data.T)
        x_min, x_max = data[:, 0].min(), data[:, 0].max()
        y_min, y_max = data[:, 1].min(), data[:, 1].max()
        x, y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([x.ravel(), y.ravel()])
        z = np.reshape(kde(positions).T, x.shape)
        plt.contour(x, y, z, levels=4, colors=color, linewidths=0.5)

    draw_contour(z0_np, color='blue')
    draw_contour(z1_np, color='red')

    plt.legend()
    plt.title(f"Bridge Transport — {direction} (iter {outer_iter_idx})")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(folder_fig, f"{direction}_iter_{outer_iter_idx}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()



def plot_moment(
    args,
    mean_gen: List[torch.Tensor],
    std_gen: List[torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
    direction: str
):
    
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    experiment_name_folder = os.path.join(args.experiment_dir,args.experiment_name)

    if not os.path.exists(experiment_name_folder):
        os.makedirs(experiment_name_folder)


    fig_folder = os.path.join(experiment_name_folder, "moment_plot")
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    mean_gen_tensor = torch.stack(mean_gen)  # [T, 2]
    std_gen_tensor = torch.stack(std_gen)    # [T, 2]
    x_axis = torch.arange(mean_gen_tensor.shape[0])

    for i, name in enumerate(['x', 'y']):
        # Mean
        plt.figure(figsize=(8, 4))
        plt.plot(x_axis, mean_gen_tensor[:, i], label="Generated", marker='o')
        plt.hlines(mean[0, i].item(), x_axis[0], x_axis[-1], colors='r', linestyles='--', label="Target")
        plt.xlabel("Outer Iteration")
        plt.ylabel("Mean")
        plt.title(f"{name.upper()} Mean over Training ({direction})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{fig_folder}/mean_{name}_{direction}.png", dpi=300)
        plt.close()

        # Std
        plt.figure(figsize=(8, 4))
        plt.plot(x_axis, std_gen_tensor[:, i], label="Generated", marker='o')
        plt.hlines(std[0, i].item(), x_axis[0], x_axis[-1], colors='r', linestyles='--', label="Target")
        plt.xlabel("Outer Iteration")
        plt.ylabel("Standard Deviation")
        plt.title(f"{name.upper()} Std over Training ({direction})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{fig_folder}/std_{name}_{direction}.png", dpi=300)
        plt.close()
