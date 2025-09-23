from accelerate import Accelerator
from datasets.datasets_registry import GaussianConfig, GaussianMixtureConfig


from accelerate import Accelerator
from datasets.datasets_registry import GaussianConfig


class ExperimentConfig:
    def __init__(self):
        # ───── Reproducibility
        self.seed = 42

        # ───── Experiment Info
        self.project_name = "DSBM_N_BRIDGES"
        self.experiment_dir = "experiments_debug"
        self.experiment_name = "test_properties_01_01"
        self.experiment_type = "latent"
        self.seed = 14

        # ───── Data Parameters
        self.dim = 2
        self.batch_size = 256
        self.n_distributions = 3
        self.separation_train_test = False
        self.nb_points_test = 1000
        self.leave_out_list = []

        # ───── Dataset Configuration
        self.distributions = DistributionConfig(dim=self.dim)

        # ───── Simulation Parameters

        self.mode_simul_train = "sde"
        self.mode_simul_inf = "sde"
        self.first_direction = "backward"
        self.coeff_sigma = 1
        self.first_coupling = "ind"
        self.sigma = 1
        self.sigma_mode = "mono"
        self.sigma_linspace = None
        self.num_simulation_steps = 40
        self.nb_inner_opt_steps = 10000
        self.nb_outer_iterations = 40
        self.eps = 1e-3
        self.loss_scale = True

        # ───── EMA Parameters

        self.ema = True
        self.decay_ema = 0.9999

        # Warmup epoch

        self.warmup = True
        self.warmup_nb_inner_opt_steps = 10000
        self.warmup_epoch = 0
        # ───── Optimization
        self.lr = 1e-3
        self.grad_clip = 10
        self.optimizer_type = "adamw"
        self.optimizer_params = {"betas": (0.9, 0.999), "weight_decay": 0.01}

        # --- Network General

        self.model_name = "mlp"

        # ───── Network: Forward score model

        self.net_fwd_layers = [256, 256]
        self.net_fwd_time_dim = 128

        # ───── Network: Backward score model
        self.net_bwd_layers = [256, 256]
        self.net_bwd_time_dim = 128

        # ----- Inference

        self.sigma_inference = self.sigma
        self.num_sample_metric = 1000

        # ───── Visualisation
        self.fps = 20

        self.param_metric = False
        self.plot_vis = True
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


class DistributionConfig:
    def __init__(self, dim: int = 2, n_samples: int = 2000):
        self.dim = dim  # In Experiment Config
        self.n_samples = 2000

        self.distributions = [
            GaussianMixtureConfig(
                time=0.0,
                means=[[-3, 4], [-3, 0]],
                stds=[[0.5, 0.5], [0.5, 0.5]],
                weights=[0.5, 0.5],
                n_samples=self.n_samples,
            ),
            GaussianMixtureConfig(
                time=1.0,
                means=[[3, 4], [3, 0]],
                stds=[[0.5, 0.5], [0.5, 0.5]],
                weights=[0.5, 0.5],
                n_samples=self.n_samples,
            ),
            GaussianMixtureConfig(
                time=2.0,
                means=[[0, -6], [0, 10]],
                stds=[[0.5, 0.5], [0.5, 0.5]],
                weights=[0.5, 0.5],
                n_samples=self.n_samples,
            ),

        ]
