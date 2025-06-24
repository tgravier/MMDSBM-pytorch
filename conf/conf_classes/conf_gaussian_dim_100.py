from accelerate import Accelerator
from datasets.datasets_registry import GaussianConfig

class DistributionConfig:
    def __init__(self, dim: int = 100, n_samples: int = 2000):
        self.dim = dim
        self.n_samples = n_samples

        # Déclarations explicites de 4 distributions avec vecteurs de dimension 100
        mean_0 = [0.0] * dim
        std_0 = [1.0] * dim

        mean_1 = [5.0] * dim
        std_1 = [1.0] * dim

        mean_2 = [-5.0] * dim
        std_2 = [1.0] * dim

        mean_3 = [float(i) for i in range(dim)]
        std_3 = [2.0] * dim

        self.distributions_train = [
            GaussianConfig(time=0, mean=mean_0, std=std_0, n_samples=self.n_samples, dim=self.dim),
            GaussianConfig(time=1, mean=mean_1, std=std_1, n_samples=self.n_samples, dim=self.dim),
            GaussianConfig(time=2, mean=mean_2, std=std_2, n_samples=self.n_samples, dim=self.dim),
            GaussianConfig(time=3, mean=mean_3, std=std_3, n_samples=self.n_samples, dim=self.dim),
        ]


class ExperimentConfig:
    def __init__(self):
        # ───── Reproducibility
        self.seed = 42

        # ───── Experiment Info
        self.project_name = "DSBM_N_BRIDGES"
        self.experiment_dir = "experiments"
        self.experiment_name = "gaussian_exp_100dim"

        # ───── Data Parameters
        self.dim = 100
        self.batch_size = 128
        self.n_distributions = 4

        # ───── Dataset Configuration
        self.distributions = DistributionConfig(dim=self.dim)

        # ───── Simulation Parameters
        self.first_coupling = "ref"
        self.sigma = 1
        self.num_simulation_steps = 20
        self.nb_inner_opt_steps = 10000
        self.nb_outer_iterations = 20
        self.eps = 1e-3

        # ───── Optimization
        self.lr = 1e-3
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

        self.debug = False