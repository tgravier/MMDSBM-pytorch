from accelerate import Accelerator
from datasets.datasets_registry import GaussianConfig, GaussianMixtureConfig


from accelerate import Accelerator
from datasets.datasets_registry import (
    GaussianConfig,
    CircleConfig,
    PhateFromTrajectoryConfig,
)


# Commentary :


class ExperimentConfig:
    def __init__(self):
        # ───── Reproducibility
        self.seed = 42

        # ───── Experiment Info
        self.project_name = "DSBM_N_BRIDGES_02"
        self.experiment_dir = "experiments"
        self.experiment_name = "phate_10"

        # ───── Data Parameters
        self.dim = 100
        self.batch_size = 256
        self.n_distributions = 10

        # ───── Dataset Configuration
        self.distributions = DistributionConfig(dim=self.dim)

        # ───── Simulation Parameters
        self.warmup_epoch = 5  ## Warmup
        self.first_coupling = "ind"
        self.sigma = 1
        self.num_simulation_steps = 40
        self.nb_inner_opt_steps = 20000
        self.nb_outer_iterations = 80
        self.eps = 1e-4

        # ───── Optimization
        self.lr = 2e-4
        self.grad_clip = 4
        self.optimizer_type = "adam"
        self.optimizer_params = {"betas": (0.9, 0.999), "weight_decay": 0.0}

        # ───── Network: Forward score model
        self.net_fwd_layers = [128, 256, 128,]
        self.net_fwd_time_dim = 64

        # ───── Network: Backward score model
        self.net_bwd_layers = [128, 256, 128,]
        self.net_bwd_time_dim = 64

        # ───── Visualisation
        self.fps = 20

        self.plot_vis = True
        self.log_wandb_traj = True
        self.plot_vis_n_epoch = 1
        self.num_sample_vis = 256
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

        self.debug = False


class DistributionConfig:
    def __init__(self, dim: int = 2, n_samples: int = 1000):
        self.dim = dim
        self.n_samples = n_samples

        self.distributions_train = [
            PhateFromTrajectoryConfig(
                time=0,
                embedding_dim=100,
                file_path="datasets/data/phate_from_trajectory/pcs_label_0.npz",
            ),

                        PhateFromTrajectoryConfig(
                time=1,
                embedding_dim=100,
                file_path="datasets/data/phate_from_trajectory/pcs_label_0.npz",
            ),

                        PhateFromTrajectoryConfig(
                time=2,
                embedding_dim=100,
                file_path="datasets/data/phate_from_trajectory/pcs_label_1.npz",
            ),


                        PhateFromTrajectoryConfig(
                time=3,
                embedding_dim=100,
                file_path="datasets/data/phate_from_trajectory/pcs_label_1.npz",
            ),
                        PhateFromTrajectoryConfig(
                time=4,
                embedding_dim=100,
                file_path="datasets/data/phate_from_trajectory/pcs_label_2.npz",
            ),
                                    PhateFromTrajectoryConfig(
                time=5,
                embedding_dim=100,
                file_path="datasets/data/phate_from_trajectory/pcs_label_2.npz",
            ),
                
                        PhateFromTrajectoryConfig(
                time=6,
                embedding_dim=100,
                file_path="datasets/data/phate_from_trajectory/pcs_label_3.npz",
            ),
                                    PhateFromTrajectoryConfig(
                time=7,
                embedding_dim=100,
                file_path="datasets/data/phate_from_trajectory/pcs_label_3.npz",
            ),
                        PhateFromTrajectoryConfig(
                time=8,
                embedding_dim=100,
                file_path="datasets/data/phate_from_trajectory/pcs_label_4.npz",
            ),

                                    PhateFromTrajectoryConfig(
                time=9,
                embedding_dim=100,
                file_path="datasets/data/phate_from_trajectory/pcs_label_4.npz",
            ),
        ]
