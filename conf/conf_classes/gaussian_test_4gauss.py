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
        self.project_name = "DSBM_N_BRIDGES"
        self.experiment_dir = "experiments_debug"
        self.experiment_name = "gaussian_4_gaussian_test_commit_warmupRF_finite_dataloader"

        # ───── Data Parameters
        self.dim = 2
        self.batch_size = 64
        self.n_distributions = 4

        # ───── Dataset Configuration
        self.distributions = DistributionConfig(dim=self.dim)

        # ───── Simulation Parameters
        self.warmup_epoch = 0
        self.first_coupling = "ind"
        self.sigma = 0.5
        self.num_simulation_steps = 80
        self.nb_inner_opt_steps = 5000
        self.nb_outer_iterations = 100
        self.eps = 1e-3

        # ───── Optimization
        self.lr = 1e-3
        self.grad_clip = 1.0
        self.optimizer_type = "adam"
        self.optimizer_params = {"betas": (0.9, 0.999), "weight_decay": 0.0}

        # ───── Network: Forward score model
        self.net_fwd_layers = [
            128,
            128,
        ]
        self.net_fwd_time_dim = 64

        # ───── Network: Backward score model
        self.net_bwd_layers = [128, 128]
        self.net_bwd_time_dim = 64

        # ───── Visualisation
        self.fps = 20
        self.plot_vis_n_epoch = 1
        self.num_sample_vis = 1000
        self.plot_traj = False
        self.number_traj = 20

        # ───── Accelerator
        self.accelerator = Accelerator()

        self.gpu_id = 2

        # ───── Debug

        self.debug = True


class DistributionConfig:
    def __init__(self, dim: int = 2, n_samples: int = 1000):
        self.dim = dim
        self.n_samples = n_samples

        self.distributions_train = [
            GaussianConfig(
                time=0,
                mean=[0, 0],
                std=[1, 1],
                n_samples=self.n_samples,
                dim=2,
            ),
            GaussianConfig(
                time=1,
                mean=[4, 0],
                std=[1, 1],
                n_samples=self.n_samples,
                dim=2,
            ),
            GaussianConfig(
                time=2,
                mean=[4, 4],
                std=[1, 1],
                n_samples=self.n_samples,
                dim=2,
            ),
                        
            GaussianConfig(
                time=3,
                mean=[0, 4],
                std=[1, 1],
                n_samples=self.n_samples,
                dim=2,
            ),
            
        ]
