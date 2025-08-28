from accelerate import Accelerator
from datasets.datasets_registry import (
    GaussianConfig,
    GaussianMixtureConfig,
    PhateFromTrajectoryConfig,
)


from accelerate import Accelerator
from datasets.datasets_registry import GaussianConfig, CircleConfig, BiotineConfig
import os


# Commentary :


class ExperimentConfig:
    def __init__(self):
        # ───── Reproducibility

        # ───── Experiment Info
        self.project_name = "DSBM_N_BRIDGES_BIOTINE"
        self.experiment_dir = "experiments_debug"
        self.experiment_name = "biotine_7d_01"
        self.experiment_type = "latent"
        self.seed = 13

        # ───── Data Parameters
        self.dim = 1024
        self.batch_size = 64
        self.n_distributions = 7
        self.separation_train_test = False
        self.nb_points_test = 1000
        self.leave_out_list = []

        # ───── Dataset Configuration
        self.distributions = DistributionConfig(dim=self.dim)

        # ───── Simulation Parameters

        self.first_direction = "backward"
        self.coeff_sigma = 1
        self.first_coupling = "ind"
        self.sigma = 0.3
        self.sigma_mode = "mono"
        self.sigma_linspace = None
        self.num_simulation_steps = 600
        self.nb_inner_opt_steps = 20000
        self.nb_outer_iterations = 40
        self.eps = 1e-3
        self.loss_scale = True

        # ───── EMA Parameters

        self.ema = True
        self.decay_ema = 0.9999

        # Warmup epoch

        self.warmup = True
        self.warmup_nb_inner_opt_steps = 100000
        self.warmup_epoch = 0
        # ───── Optimization
        self.lr = 2e-4
        self.grad_clip = 1
        self.optimizer_type = "adamw"
        self.optimizer_params = {"betas": (0.9, 0.999), "weight_decay": 0.01}

        # --- Network General

        self.model_name = "mlp_film"

        # ───── Network: Forward score model

        self.net_fwd_layers = [2048, 2048]
        self.net_fwd_time_dim = 512

        # ───── Network: Backward score model
        self.net_bwd_layers = [2048, 2048]
        self.net_bwd_time_dim = 512

        # ----- Inference

        self.sigma_inference = self.sigma
        self.num_sample_metric = 10

        # ───── Visualisation
        self.fps = 20

        self.plot_vis = False
        self.log_wandb_traj = True
        self.plot_vis_n_epoch = 1
        self.num_sample_vis = 1024
        self.plot_traj = False
        self.number_traj = 20

        # ───── Metric

        self.rescale = False

        self.log_wandb_loss = True

        self.display_swd = False
        self.log_wandb_swd = False
        self.display_swd_n_epoch = 1

        self.display_mmd = False
        self.log_wandb_mmd = False
        self.display_mmd_n_epoch = 1
        self.mmd_kernel = "rbf"  # Options: "gaussian", "laplacian", "energy", "rbf"
        self.mmd_blur = 1.0

        self.display_energy = False
        self.log_wandb_energy = False
        self.display_energy_n_epoch = 1

        # ───── Save Networks

        self.save_networks = True
        self.save_networks_n_epoch = 1

        # ------- Save Generation

        self.save_generation = True

        # ───── Accelerator
        self.accelerator = Accelerator()

        self.gpu_id = 2

        # ───── Debug

        self.debug = True


def get_file_path(base_dir, time):
    return os.path.join(base_dir, f"time_{time}.pt")


class DistributionConfig:
    def __init__(self, dim: int, n_samples: int = 2381):
        self.dim = dim

        base_dir = "/projects/static2dynamic/datasets/biotine/SD2_latent_codes"

        times = list(range(0, 19, 3))
        real_times = list(range(0, 7))

        self.distributions = [
            BiotineConfig(
            time=t,
            dim=dim,
            file_path=get_file_path(base_dir, t + 1),
            )
            for t, r_t in zip(times, real_times)
        ]
