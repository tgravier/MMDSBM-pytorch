from accelerate import Accelerator
from datasets.datasets_registry import GaussianConfig, GaussianMixtureConfig


from accelerate import Accelerator
from datasets.datasets_registry import GaussianConfig, CircleConfig


# Commentary :
class ExperimentConfig:
    def __init__(self):
        # ───── Reproducibility
        self.seed = 42

        # ───── Experiment Info
        self.project_name = "DSBM_N_BRIDGES_02"
        self.experiment_dir = "experiments_debug"
        self.experiment_name = "gaussian_dim2_02"

        # ───── Data Parameters
        self.dim = 2
        self.batch_size = 64
        self.n_distributions = 4

        # ───── Dataset Configuration
        self.distributions = DistributionConfig(dim=self.dim)

        # ───── Simulation Parameters
        self.warmup_epoch = 5  ## Warmup
        self.first_coupling = "ind"
        self.sigma = 1
        self.num_simulation_steps = 80
        self.nb_inner_opt_steps = 1000
        self.nb_outer_iterations = 20
        self.eps = 1e-4

        # ───── Optimization
        self.lr = 1e-3
        self.grad_clip = 1.0
        self.optimizer_type = "adam"
        self.optimizer_params = {"betas": (0.9, 0.999), "weight_decay": 0.0}


        # ───── Network: Forward score model
        self.net_fwd_layers = [128, 128]
        self.net_fwd_time_dim = 64

        # ───── Network: Backward score model
        self.net_bwd_layers = [128,128]
        self.net_bwd_time_dim = 128

        # ───── Visualisation
        self.fps = 20

        self.plot_vis = True
        self.log_wandb_traj = True
        self.plot_vis_n_epoch = 5
        self.num_sample_vis = 1000
        self.plot_traj = False
        self.number_traj = 20

        # ───── Metric

        self.log_wandb_loss = True

        self.display_swd = True
        self.log_wandb_swd = True
        self.display_swd_n_epoch = 1

        self.display_mmd = True
        self.log_wandb_mmd = True
        self.display_mmd_n_epoch = 1
        self.mmd_kernel = "rbf"  # Options: "gaussian", "laplacian", "energy", "rbf"
        self.mmd_blur = 1.0

        self.display_energy = False
        self.log_wandb_energy = True
        self.display_energy_n_epoch = 1

        # ───── Save Networks

        self.save_networks = True
        self.save_networks_n_epoch = 1

        # ───── Accelerator
        self.accelerator = Accelerator()

        self.gpu_id = 2

        # ───── Debug

        self.debug = True

class ExperimentConfig:
    def __init__(self):
        # ───── Reproducibility
        self.seed = 42

        # ───── Experiment Info
        self.project_name = "DSBM_N_BRIDGES"
        self.experiment_dir = "experiments_debug"
        self.experiment_name = "debug_warmup_test_commit_circle"

        # ───── Data Parameters
        self.dim = 2
        self.batch_size = 64
        self.n_distributions = 3

        # ───── Dataset Configuration
        self.distributions = DistributionConfig(dim=self.dim)

        # ───── Simulation Parameters
        self.warmup_epoch = 20
        self.first_coupling = "ind"
        self.sigma = 0.5
        self.num_simulation_steps = 80
        self.nb_inner_opt_steps = 1000
        self.nb_outer_iterations = 100
        self.eps = 1e-3

        # ───── Optimization
        self.lr = 1e-3
        self.grad_clip = 1.0
        self.optimizer_type = "adam"
        self.optimizer_params = {"betas": (0.9, 0.999), "weight_decay": 0.0}

        # ───── Network: Forward score model
        self.net_fwd_layers = [128, 128, 128]
        self.net_fwd_time_dim = 128

        # ───── Network: Backward score model
        self.net_bwd_layers = [128, 128, 128]
        self.net_bwd_time_dim = 128

        # ───── Visualisation
        self.fps = 20
        self.plot_vis_n_epoch = 5
        self.num_sample_vis = 1000
        self.plot_traj = False
        self.number_traj = 20

        # ───── Accelerator
        self.accelerator = Accelerator()

        self.gpu_id = 2

        # ───── Debug

        self.debug = True


class DistributionConfig:
    def __init__(self, dim: int = 2, n_samples: int = 4000):
        self.dim = dim  # In Experiment Config
        self.n_samples = 1000

        self.distributions_train = [
            CircleConfig(
                time=0,
                n_samples=self.n_samples,
                center=[0, 0],
                radius=1,
                thickness=0.05,
            ),
            CircleConfig(
                time=1,
                n_samples=self.n_samples,
                center=[0, 0],
                radius=3,
                thickness=0.05,
            ),
            CircleConfig(
                time=2,
                n_samples=self.n_samples,
                center=[0, 0],
                radius=1,
                thickness=0.05,
            ),
        ]
