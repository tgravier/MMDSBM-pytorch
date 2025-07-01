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
        self.experiment_name = "test_warmup_02"

        # ───── Data Parameters
        self.dim = 2
        self.batch_size = 256
        self.n_distributions = 3

        # ───── Dataset Configuration
        self.distributions = DistributionConfig(dim=self.dim)

        # ───── Simulation Parameters
        self.first_coupling = "ref"
        self.sigma = 1
        self.warmup_epoch = 3
        self.num_simulation_steps = 100
        self.nb_inner_opt_steps = 1000
        self.nb_outer_iterations = 100
        self.eps = 1e-3

        # ───── Optimization
        self.lr = 5e-4
        self.grad_clip = 1.0
        self.optimizer_type = "adam"
        self.optimizer_params = {"betas": (0.9, 0.999), "weight_decay": 0.0}

        # ───── Network: Forward score model
        self.net_fwd_layers = [128, 256, 128]
        self.net_fwd_time_dim = 128

        # ───── Network: Backward score model
        self.net_bwd_layers = [128, 256, 128]
        self.net_bwd_time_dim = 128
        

        # ───── Visualisation
        self.plot_vis_n_epoch = 3
        self.num_sample_vis = 1500
        self.fps = 20
        self.plot_traj = False
        self.number_traj = 0

        # ───── Accelerator
        self.accelerator = Accelerator()

        self.gpu_id = 2

        # ───── Debug

        self.debug = True


class DistributionConfig:
    def __init__(self, dim: int = 2, n_samples: int = 2000):
        self.dim = dim  # In Experiment Config
        self.n_samples = 2000

        self.distributions_train = [
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
