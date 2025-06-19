#utils/visualisation.py

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
    t_pairs,
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
            t_pairs = t_pairs,
            direction_tosample=direction,
            N=num_steps,
            sig=sigma,
            device=args.accelerator.device
        )[0][-1]
    elif direction == 'backward':
        generated_samples = sample_sde(
            zstart=z1_sampled,
            net_dict=model.net_dict,
            t_pairs = t_pairs,
            direction_tosample=direction,
            N=num_steps,
            sig=sigma,
            device=args.accelerator.device
        )[0][-1]

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

    save_path = os.path.join(folder_fig, f"iter_{outer_iter_idx}_{direction}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()




@torch.no_grad()
def make_trajectory_gif(
    args,
    direction_tosample: str,
    net_dict: dict,
    dataset_train: list,
    outer_iter_idx: int,
    num_samples: int = 100,
    num_steps: int = 200,
    sigma: float = 1.0,
    fps: int = 20,
    one_bridge: bool = False
):  
    experiment_name_folder = os.path.join(args.experiment_dir, args.experiment_name)

    folder_traj = os.path.join(
        experiment_name_folder,
        "traj" if one_bridge else "traj_bridges"
    )
    os.makedirs(folder_traj, exist_ok=True)

    device = next(net_dict[direction_tosample].parameters()).device

    # === Start points ===
    if direction_tosample == "forward":
        z0 = dataset_train[0].get_all().to(device)
    elif direction_tosample == "backward":
        z0 = dataset_train[-1].get_all().to(device)
    else:
        raise ValueError("Invalid direction")

    idx = torch.randint(0, z0.shape[0], (num_samples,))
    z0_sampled = z0[idx]

    t_pairs = [dataset_train[0].get_time(), dataset_train[-1].get_time()]

    if not one_bridge:
        num_steps = num_steps * (len(dataset_train) - 1)

    # === Simulate trajectory
    generated, time = sample_sde(
        zstart=z0_sampled,
        net_dict=net_dict,
        t_pairs=t_pairs,
        direction_tosample=direction_tosample,
        N=num_steps,
        sig=sigma,
        device=device
    )

    generated = [g.cpu().numpy() for g in generated]  # list of [B, D]
    time = [float(t) for t in time]  # ensure it's a list of floats

    # === Load fixed distributions
    cmap = cm.get_cmap('tab10', len(dataset_train))
    distrib_data = []
    for i, ds in enumerate(dataset_train):
        data = ds.get_all().cpu().numpy()
        distrib_data.append((ds.get_time(), data, cmap(i)))

    # === Plot bounds
    all_points = np.concatenate([d for _, d, _ in distrib_data], axis=0)
    x_min, x_max = all_points[:, 0].min() - 0.5, all_points[:, 0].max() + 0.5
    y_min, y_max = all_points[:, 1].min() - 0.5, all_points[:, 1].max() + 0.5

    # === Init plot
    fig, ax = plt.subplots(figsize=(6, 6))
    scatters = []

    # Plot reference distributions
    for ti, data, color in distrib_data:
        s = ax.scatter(data[:, 0], data[:, 1], s=30, alpha=0.3, color=color, label=f"t={ti:.2f}")
        scatters.append(s)

    # Init for moving particles
    gen_scat = ax.scatter([], [], s=5, color='green', label="Generated")
    scatters.append(gen_scat)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper right', fontsize=6)
    ax.grid(True)

    # === Animation update function (affiche t réel)
    def update(frame):
        z = generated[frame]
        gen_scat.set_offsets(z[:, :2])
        ax.set_title(f"Bridge Evolution — Epoch {outer_iter_idx} — t = {time[frame]:.3f}")
        return scatters

    # === Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(generated),
        interval=1000 // fps,
        blit=True
    )

    save_path = os.path.join(
        folder_traj,
        f"traj_iter_{outer_iter_idx}_tpairs_{t_pairs}_direction_{direction_tosample}.gif"
    )
    anim.save(save_path, writer='pillow', fps=fps)




@torch.no_grad()
def make_trajectory_old(
    args,
    direction_tosample: str,
    net_dict: dict,
    dataset_train: list,
    outer_iter_idx: int,
    num_samples: int = 100,
    num_steps: int = 200,
    sigma: float = 1.0,
    one_bridge: bool = False
):
    experiment_name_folder = os.path.join(args.experiment_dir, args.experiment_name)
    folder_traj = os.path.join(
        experiment_name_folder,
        "traj" if one_bridge else "traj_bridges"
    )
    os.makedirs(folder_traj, exist_ok=True)

    device = next(net_dict[direction_tosample].parameters()).device

    # === Start points
    if direction_tosample == "forward":
        z0 = dataset_train[0].get_all().to(device)
    elif direction_tosample == "backward":
        z0 = dataset_train[-1].get_all().to(device)
    else:
        raise ValueError("Invalid direction")

    idx = torch.randint(0, z0.shape[0], (num_samples,))
    z0_sampled = z0[idx]

    t_pairs = [dataset_train[0].get_time(), dataset_train[-1].get_time()]

    if not one_bridge:
        num_steps = num_steps * (len(dataset_train) - 1)

    # === Simulate trajectory
    from bridge.sde.bridge_sampler import sample_sde
    generated, time = sample_sde(
        zstart=z0_sampled,
        net_dict=net_dict,
        t_pairs=t_pairs,
        direction_tosample=direction_tosample,
        N=num_steps,
        sig=sigma,
        device=device
    )

    generated = [g.cpu().numpy() for g in generated]
    time = [float(t) for t in time]

    # === Load fixed distributions
    cmap = cm.get_cmap('tab10', len(dataset_train))
    distrib_data = []
    for i, ds in enumerate(dataset_train):
        data = ds.get_all().cpu().numpy()
        distrib_data.append((ds.get_time(), data, cmap(i)))

    # === Plot bounds
    all_points = np.concatenate([d for _, d, _ in distrib_data], axis=0)
    x_min, x_max = all_points[:, 0].min() - 0.5, all_points[:, 0].max() + 0.5
    y_min, y_max = all_points[:, 1].min() - 0.5, all_points[:, 1].max() + 0.5

    # === Create grid for score norm visualization
    resolution = 100  # e.g. 100x100 grid
    x_grid = torch.linspace(x_min, x_max, resolution)
    y_grid = torch.linspace(y_min, y_max, resolution)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    grid_points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device)

    score_model = net_dict[direction_tosample].eval()

    fig, ax = plt.subplots(figsize=(6, 6))

    # Display norm of score
    norm_image = ax.imshow(
        np.zeros((resolution, resolution)),
        extent=[x_min, x_max, y_min, y_max],
        origin='lower',
        cmap='inferno',
        alpha=0.5
    )

    # Plot distributions
    scatters = []
    for ti, data, color in distrib_data:
        s = ax.scatter(data[:, 0], data[:, 1], s=30, alpha=0.3, color=color, label=f"t={ti:.2f}")
        scatters.append(s)

    gen_scat = ax.scatter([], [], s=5, color='green', label="Generated")
    scatters.append(gen_scat)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper right', fontsize=6)
    ax.grid(True)

    # === Set video FPS based on dt_i
    if len(time) < 2:
        raise ValueError("Not enough time steps to compute dt.")
    dt_i = abs(time[2] - time[1])
    video_fps = 1.0 / dt_i

    save_path = os.path.join(
        folder_traj,
        f"traj_iter_{outer_iter_idx}_tpairs_{t_pairs}_direction_{direction_tosample}.mp4"
    )
    writer = FFMpegWriter(fps=video_fps)

    with writer.saving(fig, save_path, dpi=200):
        for i in range(len(generated)):
            t_val = torch.full((grid_points.size(0), 1), time[i], device=device)
            with torch.no_grad():
                score_vec = score_model(grid_points, t_val)  # [N, 2]
                score_norm = score_vec.norm(dim=1).cpu().numpy()
                score_grid = score_norm.reshape((resolution, resolution))

            norm_image.set_data(score_grid)
            norm_image.set_clim(0, score_grid.max())

            z = generated[i]
            gen_scat.set_offsets(z[:, :2])
            ax.set_title(f"Bridge Evolution — Epoch {outer_iter_idx} — t = {time[i]:.3f}")
            writer.grab_frame()

    plt.close(fig)

@torch.no_grad()
def make_trajectory(
    args,
    direction_tosample: str,
    net_dict: dict,
    dataset_train: list,
    outer_iter_idx: int,
    num_samples: int = 100,
    num_steps: int = 200,
    sigma: float = 1.0,
    one_bridge: bool = False
):
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.animation import FFMpegWriter

    experiment_name_folder = os.path.join(args.experiment_dir, args.experiment_name)
    folder_traj = os.path.join(
        experiment_name_folder,
        "traj" if one_bridge else "traj_bridges"
    )
    os.makedirs(folder_traj, exist_ok=True)

    device = next(net_dict[direction_tosample].parameters()).device

    # === Initial sample points
    if direction_tosample == "forward":
        z0 = dataset_train[0].get_all().to(device)
    elif direction_tosample == "backward":
        z0 = dataset_train[-1].get_all().to(device)
    else:
        raise ValueError("Invalid direction")

    idx = torch.randint(0, z0.shape[0], (num_samples,))
    z0_sampled = z0[idx]

    t_pairs = [dataset_train[0].get_time(), dataset_train[-1].get_time()]
    if not one_bridge:
        num_steps = num_steps * (len(dataset_train) - 1)

    # === Simulate trajectory
    from bridge.sde.bridge_sampler import inference_sample_sde
    generated, time = inference_sample_sde(
        zstart=z0_sampled,
        net_dict=net_dict,
        t_pairs=t_pairs,
        direction_tosample=direction_tosample,
        N=num_steps,
        sig=sigma,
        device=device
    )

    generated = [g.cpu().numpy() for g in generated]
    time = [float(t) for t in time]

    # === Load dataset distributions
    cmap = cm.get_cmap('tab10', len(dataset_train))
    distrib_data = []
    for i, ds in enumerate(dataset_train):
        data = ds.get_all().cpu().numpy()
        distrib_data.append((ds.get_time(), data, cmap(i)))

    all_points = np.concatenate([d for _, d, _ in distrib_data], axis=0)
    x_min, x_max = all_points[:, 0].min() - 0.5, all_points[:, 0].max() + 0.5
    y_min, y_max = all_points[:, 1].min() - 0.5, all_points[:, 1].max() + 0.5

    # === Grid for score
    resolution = 50
    stride = 3           # show 1 arrow every 3 grid points
    arrow_scale = 0.5    # visual scale of arrows

    x_grid = torch.linspace(x_min, x_max, resolution)
    y_grid = torch.linspace(y_min, y_max, resolution)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    grid_points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device)

    score_model = net_dict[direction_tosample].eval()

    fig, ax = plt.subplots(figsize=(6, 6))

    norm_image = ax.imshow(
        np.zeros((resolution, resolution)),
        extent=[x_min, x_max, y_min, y_max],
        origin='lower',
        cmap='inferno',
        alpha=0.5
    )

    # === Initial quiver (empty)
    quiver = ax.quiver(
        X[::stride, ::stride].cpu().numpy(), Y[::stride, ::stride].cpu().numpy(),
        np.zeros_like(X[::stride, ::stride].cpu().numpy()),
        np.zeros_like(Y[::stride, ::stride].cpu().numpy()),
        color='cyan',
        angles='xy',
        scale_units='xy',
        scale=1,
        width=0.003
    )

    for ti, data, color in distrib_data:
        ax.scatter(data[:, 0], data[:, 1], s=30, alpha=0.3, color=color, label=f"t={ti:.2f}")

    gen_scat = ax.scatter([], [], s=5, color='green', label="Generated")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper right', fontsize=6)
    ax.grid(True)

    if len(time) < 2:
        raise ValueError("Not enough time steps to compute dt.")
    dt_i = abs(time[2] - time[1])
    video_fps = 1.0 / dt_i

    save_path = os.path.join(
        folder_traj,
        f"traj_iter_{outer_iter_idx}_tpairs_{t_pairs}_direction_{direction_tosample}.mp4"
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
            quiver.set_UVC(
                U_scaled[::stride, ::stride],
                V_scaled[::stride, ::stride]
            )

            z = generated[i]
            gen_scat.set_offsets(z[:, :2])
            ax.set_title(f"Bridge Evolution — Epoch {outer_iter_idx} — t = {time[i]:.3f}")
            writer.grab_frame()

    plt.close(fig)

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




