from accelerate import Accelerator
from datasets.datasets_registry import GaussianConfig

class DistributionConfig:
    def __init__(self, dim: int = 2, n_samples: int = 2000):
        self.dim = dim
        self.n_samples = n_samples

        # ───── Define training distributions (3 Gaussians)
        self.distributions_train = [
            GaussianConfig(time=0, mean=[0, 0], std=[1, 1], n_samples=self.n_samples, dim=self.dim),
            GaussianConfig(time=1, mean=[5, 5], std=[1, 1], n_samples=self.n_samples, dim=self.dim),
            GaussianConfig(time=2, mean=[0, 0], std=[0.001, 0.001], n_samples=self.n_samples, dim=self.dim),
        ]

        

class ExperimentConfig:
    def __init__(self):
        # ───── Reproducibility
        self.seed = 42

        # ───── Experiment Info
        self.experiment_dir = "experiments_debug"
        self.experiment_name = "tiny_gaussian_01"

        # ───── Data Parameters
        self.dim = 2
        self.batch_size = 128
        self.n_distributions = 3

        # ───── Dataset Configuration
        self.distributions = DistributionConfig(dim=self.dim)

        # ───── Simulation Parameters
        self.first_coupling = "ref"
        self.sigma = 1
        self.num_simulation_steps = 80
        self.nb_inner_opt_steps = 20000
        self.nb_outer_iterations = 20
        self.eps = 1e-3

        # ───── Optimization
        self.lr = 1e-4
        self.grad_clip = 1.0
        self.optimizer_type = "adam"
        self.optimizer_params = {
            "betas": (0.9, 0.999),
            "weight_decay": 0.0
        }

        # ───── Network: Forward score model
        self.net_fwd_layers = [128, 128]
        self.net_fwd_time_dim = 64

        # ───── Network: Backward score model
        self.net_bwd_layers = [128, 128]
        self.net_bwd_time_dim = 64

        # ───── Logging / Debug
        self.vis_every = 1

        # ───── Accelerator
        self.accelerator = Accelerator()
