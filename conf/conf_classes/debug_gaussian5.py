from accelerate import Accelerator
from datasets.datasets_registry import GaussianConfig


class ExperimentConfig:
    def __init__(self):
        # ───── Reproducibility
        self.seed = 42

        # ───── Experiment Info
        self.project_name = "DSBM_N_BRIDGES"
        self.experiment_dir = "experiments_debug"
        self.experiment_name = "debug_commit_loss_wass_01_ind_finiteloader_02"

        # ───── Data Parameters
        self.dim = 2
        self.batch_size = 512
        self.n_distributions = 4

        # ───── Dataset Configuration
        self.distributions = DistributionConfig(dim=self.dim)

        # ───── Simulation Parameters
        self.first_coupling = "ind"
        self.sigma = 1
        self.num_simulation_steps = 80
        self.nb_inner_opt_steps = 20000
        self.nb_outer_iterations = 20
        self.eps = 1e-3

        # Warmup epoch

        self.warmup = True

        # ───── Optimization
        self.lr = 1e-4
        self.grad_clip = 1.0
        self.optimizer_type = "adam"
        self.optimizer_params = {"betas": (0.9, 0.999), "weight_decay": 0.0}

        # ───── Network: Forward score model
        self.net_fwd_layers = [128, 128]
        self.net_fwd_time_dim = 64

        # ───── Network: Backward score model
        self.net_bwd_layers = [128, 128]
        self.net_bwd_time_dim = 64

        # ───── Logging
        self.vis_every = 1

        
        # ----- Inference

        self.sigma_inference = 0

        # ───── Visualisation
        self.fps = 20

        self.plot_vis = True
        self.log_wandb_traj = True
        self.plot_vis_n_epoch = 1
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





class DistributionConfig:
    def __init__(self, dim: int = 2, n_samples: int = 2000):
        self.dim = dim  # In Experiment Config
        self.n_samples = 10000

        # ───── Define training distributions (3 Gaussians)
        self.distributions_train = [
            GaussianConfig(
                time=0,
                mean=[5, 1],
                std=[1, 1],
                n_samples=self.n_samples,
                dim=self.dim,
            ),
            GaussianConfig(
                time=1,
                mean=[9, 5],
                std=[1, 1],
                n_samples=self.n_samples,
                dim=self.dim,
            ),
            GaussianConfig(
                time=2,
                mean=[5, 9],
                std=[1, 1],
                n_samples=self.n_samples,
                dim=self.dim,
            ),
            GaussianConfig(
                time=3,
                mean=[1, 5],
                std=[1, 1],
                n_samples=self.n_samples,
                dim=self.dim,
            ),
        ]
