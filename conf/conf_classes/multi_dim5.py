from accelerate import Accelerator
from datasets.datasets_registry import MultiConfig


class ExperimentConfig:
    def __init__(self):
        # ───── Reproducibility
        self.seed = 42

        # ───── Experiment Info
        self.project_name = "DSBM_N_BRIDGES_MULTI"
        self.experiment_dir = "experiments"
        self.experiment_name = "multi_dim5_sigma0.1"
        self.experiment_type = ""

        # ───── Data Parameters
        self.dim = 5
        self.batch_size = 1024
        self.n_distributions = 4
        self.separation_train_test = True
        self.nb_points_test = 1000
        self.recreate_dataset = False
        self.max_pairs = 100_000
        self.leave_out_list = []

        # ───── Dataset Configuration
        self.distributions = DistributionConfig(dim=self.dim)

        # ───── Simulation Parameters
        self.chunk_size = 20_000
        self.first_direction = "backward"
        self.coeff_sigma = 1
        self.first_coupling = "ind"
        self.sigma = 0.1
        self.sigma_mode = "mono"  # "mono", "multi", "multi_dim"
        self.sigma_linspace = None
        self.num_simulation_steps = 150
        self.nb_inner_opt_steps = 10_000
        self.nb_outer_iterations = 50
        self.eps = 1e-3
        self.loss_scale = True

        # ───── EMA Parameters
        self.ema = True
        self.decay_ema = 0.9999

        # ───── Warmup
        self.warmup = True
        self.warmup_nb_inner_opt_steps = 2_000
        self.warmup_epoch = 0

        # ───── Optimization
        self.lr = 1e-4
        self.grad_clip = 1
        self.optimizer_type = "adamw"
        self.optimizer_params = {"betas": (0.9, 0.999), "weight_decay": 0.01}

        # ───── Network General
        self.model_name = "mlp"

        # ───── Network: Forward score model
        self.net_fwd_layers = [256, 256]
        self.net_fwd_time_dim = 128

        # ───── Network: Backward score model
        self.net_bwd_layers = [256, 256]
        self.net_bwd_time_dim = 128

        # ───── Inference
        self.sigma_inference = self.sigma
        self.num_sample_metric = 1000

        # ───── Visualisation
        self.fps = 20
        self.plot_vis = True
        self.log_wandb_traj = True
        self.plot_vis_n_epoch = 1
        self.num_sample_vis = 1024
        self.plot_traj = False
        self.number_traj = 20

        # ───── Metrics
        self.rescale = False

        self.log_wandb_loss = True

        self.display_swd = True
        self.log_wandb_swd = True
        self.display_swd_n_epoch = 1

        self.display_mmd = True
        self.log_wandb_mmd = True
        self.display_mmd_n_epoch = 1
        self.mmd_kernel = "rbf"
        self.mmd_blur = 1.0

        # W1 metric (used in MSBM paper for MULTI benchmark)
        self.compute_w1 = True

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
        self.gpu_id = 0

        # ───── Debug
        self.debug = False


class DistributionConfig:
    def __init__(self, dim: int):
        self.dim = dim
        base_dir = "/projects/static2dynamic/datasets/Multiome/dim5"

        self.distributions = [
            MultiConfig(
                time=0,
                dim=dim,
                file_path=f"{base_dir}/time_0.npz",
            ),
            MultiConfig(
                time=1,
                dim=dim,
                file_path=f"{base_dir}/time_1.npz",
            ),
            MultiConfig(
                time=2,
                dim=dim,
                file_path=f"{base_dir}/time_2.npz",
            ),
            MultiConfig(
                time=3,
                dim=dim,
                file_path=f"{base_dir}/time_3.npz",
            ),
        ]
