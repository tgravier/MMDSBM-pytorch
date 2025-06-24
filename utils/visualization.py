# utils/visualisation.py

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
from torch import Tensor
from scipy.stats import gaussian_kde

from matplotlib import animation, cm


from bridge.sde.bridge_sampler import inference_sample_sde
from utils.metric import get_classic_metrics

from typing import List


from matplotlib.animation import FFMpegWriter


from matplotlib import cm


@torch.no_grad()
def make_trajectory(
    args,
    net_dict,
    generated,
    time,
    dataset_train,
    direction_tosample,
    outer_iter_idx,
    fps=None,
    one_bridge=False,
):
    experiment_name_folder = os.path.join(args.experiment_dir, args.experiment_name)
    folder_traj = os.path.join(
        experiment_name_folder, "traj" if one_bridge else "traj_bridges"
    )
    os.makedirs(folder_traj, exist_ok=True)

    device = next(net_dict[direction_tosample].parameters()).device

    t_pairs = [dataset_train[0].get_time(), dataset_train[-1].get_time()]

    generated = [g.cpu().numpy() for g in generated]
    time = [float(t) for t in time]

    # === Load dataset distributions
    cmap = cm.get_cmap("tab10", len(dataset_train))
    distrib_data = []
    for i, ds in enumerate(dataset_train):
        data = ds.get_all().cpu().numpy()
        distrib_data.append((ds.get_time(), data, cmap(i)))

    all_points = np.concatenate([d for _, d, _ in distrib_data], axis=0)
    x_min, x_max = all_points[:, 0].min() - 0.5, all_points[:, 0].max() + 0.5
    y_min, y_max = all_points[:, 1].min() - 0.5, all_points[:, 1].max() + 0.5

    # === Grid for score
    resolution = 50
    stride = 3  # show 1 arrow every 3 grid points
    arrow_scale = 0.5  # visual scale of arrows

    x_grid = torch.linspace(x_min, x_max, resolution)
    y_grid = torch.linspace(y_min, y_max, resolution)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing="ij")
    grid_points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device)

    score_model = net_dict[direction_tosample].eval()

    fig, ax = plt.subplots(figsize=(6, 6))

    norm_image = ax.imshow(
        np.zeros((resolution, resolution)),
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap="inferno",
        alpha=0.5,
    )

    # === Initial quiver (empty)
    quiver = ax.quiver(
        X[::stride, ::stride].cpu().numpy(),
        Y[::stride, ::stride].cpu().numpy(),
        np.zeros_like(X[::stride, ::stride].cpu().numpy()),
        np.zeros_like(Y[::stride, ::stride].cpu().numpy()),
        color="cyan",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.003,
    )

    for ti, data, color in distrib_data:
        ax.scatter(
            data[:, 0], data[:, 1], s=30, alpha=0.3, color=color, label=f"t={ti:.2f}"
        )

    gen_scat = ax.scatter([], [], s=5, color="green", label="Generated")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(True)

    if len(time) < 2:
        raise ValueError("Not enough time steps to compute dt.")
    dt_i = abs(time[2] - time[1])
    video_fps = 1.0 / dt_i

    if fps is not None:
        video_fps = fps

    save_path = os.path.join(
        folder_traj,
        f"traj_iter_{outer_iter_idx}_tpairs_{t_pairs}_direction_{direction_tosample}.mp4",
    )
    writer = FFMpegWriter(fps=video_fps)

    # === Animate trajectory and vector field
    with writer.saving(fig, save_path, dpi=200):
        for i in range(len(generated)):
            t_val = torch.full((grid_points.size(0), 1), time[i], device=device)
            with torch.no_grad():
                score_vec = score_model(grid_points, t_val)  # [N, 2]

            score_norm = score_vec.norm(dim=1).cpu().numpy()
            score_grid = score_norm.reshape((resolution, resolution))
            norm_image.set_data(score_grid)
            norm_image.set_clim(0, score_grid.max())

            # === Compute normalized vector field
            U = score_vec[:, 0].cpu().numpy().reshape((resolution, resolution))
            V = score_vec[:, 1].cpu().numpy().reshape((resolution, resolution))
            mag = np.sqrt(U**2 + V**2) + 1e-8
            U_scaled = (U / mag) * arrow_scale
            V_scaled = (V / mag) * arrow_scale

            # === Update arrows (sparse view)
            quiver.set_UVC(U_scaled[::stride, ::stride], V_scaled[::stride, ::stride])

            z = generated[i]
            gen_scat.set_offsets(z[:, :2])
            ax.set_title(
                f"Bridge Evolution — Epoch {outer_iter_idx} — t = {time[i]:.3f}"
            )
            writer.grab_frame()

    plt.close(fig)
