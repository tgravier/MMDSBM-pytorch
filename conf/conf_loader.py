import importlib
import importlib.util
import inspect
import shutil
from pathlib import Path
from typing import Optional


def load_config(
    config_name: str = None, experiment_path: str = None, resume_train: bool = False, inference : bool = False
):
    """
    Load configuration either from conf.conf_classes.<config_name> (normal mode)
    or from a saved config file inside experiment_path/conf/ (resume mode).
    """
    if resume_train or inference:
        # Load config from saved experiment folder
        exp_path = Path(experiment_path)
        conf_dir = exp_path / "conf"
        py_files = list(conf_dir.glob("*.py"))

        if not py_files:
            raise FileNotFoundError(f"No config .py file found in {conf_dir}")

        config_file = py_files[0]  # Assume only one config file per experiment
        spec = importlib.util.spec_from_file_location("loaded_config", config_file)
        loaded_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded_module)

        DistributionConfigCls = getattr(loaded_module, "DistributionConfig")
        ExperimentConfigCls = getattr(loaded_module, "ExperimentConfig")
    else:
        # Load config from module by name
        module_path = f"conf.conf_classes.{config_name}"
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            raise ImportError(
                f"Could not import configuration module '{module_path}': {e}"
            )

        DistributionConfigCls = getattr(module, "DistributionConfig")
        ExperimentConfigCls = getattr(module, "ExperimentConfig")

    # Initialize configs

    experiment_cfg = ExperimentConfigCls()
    distribution_cfg = experiment_cfg.distributions
    experiment_cfg.distribution_cfg = distribution_cfg

    exp_root = Path(experiment_cfg.experiment_dir)
    exp_path = exp_root / experiment_cfg.experiment_name
    conf_dir = exp_path / "conf"
    experiments_debug = "experiments_debug"

    if not experiment_cfg.debug:
        if str(exp_root) == experiments_debug:
            raise ValueError(
                f"Experiment dir '{exp_root}' is a debug directory but debug mode is off."
            )
        if exp_path.exists():
            if not resume_train:
                raise FileExistsError(
                    f"Experiment '{experiment_cfg.experiment_name}' already exists. "
                    f"Use --resume_train or remove the folder."
                )
        else:
            conf_dir.mkdir(parents=True, exist_ok=False)
    else:
        if str(exp_root) != experiments_debug:
            raise ValueError(
                f"Debug mode requires experiment_dir='{experiments_debug}'"
            )
        conf_dir.mkdir(parents=True, exist_ok=True)

    # Always copy config file if not in resume mode (even in debug mode)
    if not resume_train:
        src_path = Path(module.__file__)
        dst_path = conf_dir / f"{config_name}.py"
        shutil.copy2(src_path, dst_path)
        print(f"Copied config to {dst_path}")

    # Print summary
    lines = []
    lines.append("Loaded Experiment Configuration:")
    lines.append("-" * 35)
    for name, value in inspect.getmembers(experiment_cfg):
        if name.startswith("_") or inspect.ismethod(value) or inspect.isfunction(value):
            continue
        if name == "distributions":
            lines.append(
                f"{name:20}: DistributionConfig with {len(distribution_cfg.distributions)} distributions"
            )
        else:
            lines.append(f"{name:20}: {value}")
    lines.append("-" * 35)

    lines.append("Loaded Distribution Bridges:")
    lines.append("-" * 35)
    for dist in distribution_cfg.distributions:
        dist_type = type(dist).__name__
        dist_params = {
            k: v
            for k, v in vars(dist).items()
            if not k.startswith("_") and not callable(v)
        }
        param_str = ", ".join(f"{k}={v}" for k, v in dist_params.items())
        lines.append(f"{dist_type:15} | {param_str}")
    lines.append("-" * 35)

    print("\n" + "\n".join(lines) + "\n")

    summary_path = conf_dir / f"{experiment_cfg.experiment_name}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    return experiment_cfg


def export_config_dict(config_classes):
    experiment_cfg, distribution_cfg = config_classes, config_classes.distribution_cfg
    config_dict = {}

    for name, value in vars(experiment_cfg).items():
        if name.startswith("_") or name in {"distribution_cfg", "accelerator"}:
            continue
        if callable(value):
            continue
        config_dict[name] = value

    config_dict["distributions"] = [
        {
            "type": type(dist).__name__,
            "time": dist.time,
            **dist.params,
        }
        for dist in distribution_cfg.distributions
    ]
    return config_dict


def get_experiment_parameters(experiment_path: str):
    """
    Récupère les paramètres clés depuis un dossier d'expérience.

    Args:
        experiment_path: Chemin vers le dossier d'expérience (qui contient un sous-dossier conf/)

    Returns:
        dict: Dictionnaire contenant:
            - experiment_folder: Chemin du dossier d'expérience
            - sigma_inference: Sigma pour l'inférence
            - sigma_train: Sigma pour l'entraînement
            - num_simulation_steps: Nombre d'étapes de simulation
            - model_name: Nom du modèle
            - net_fwd_layers: Configuration des couches forward
            - net_bwd_layers: Configuration des couches backward
            - net_fwd_time_dim: Dimension temporelle forward
            - net_bwd_time_dim: Dimension temporelle backward
            - leave_out_list: Liste des indices de distributions à exclure
    """
    exp_path = Path(experiment_path)
    conf_dir = exp_path / "conf"

    # Vérifier que le dossier conf existe
    if not conf_dir.exists():
        raise FileNotFoundError(f"Le dossier conf/ n'existe pas dans {exp_path}")

    # Chercher le fichier de configuration Python
    py_files = list(conf_dir.glob("*.py"))
    if not py_files:
        raise FileNotFoundError(f"Aucun fichier .py trouvé dans {conf_dir}")

    # Charger le module de configuration
    config_file = py_files[0]
    spec = importlib.util.spec_from_file_location("loaded_config", config_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger le module depuis {config_file}")

    loaded_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded_module)

    # Récupérer la classe ExperimentConfig
    ExperimentConfigCls = getattr(loaded_module, "ExperimentConfig")
    experiment_cfg = ExperimentConfigCls()

    # Extraire les paramètres demandés
    params = {
        "experiment_folder": str(exp_path),
        "sigma_inference": getattr(experiment_cfg, "sigma_inference", None),
        "sigma_train": getattr(experiment_cfg, "sigma", None),
        "num_simulation_steps": getattr(experiment_cfg, "num_simulation_steps", None),
        "model_name": getattr(experiment_cfg, "model_name", None),
        "net_fwd_layers": getattr(experiment_cfg, "net_fwd_layers", None),
        "net_bwd_layers": getattr(experiment_cfg, "net_bwd_layers", None),
        "net_fwd_time_dim": getattr(experiment_cfg, "net_fwd_time_dim", None),
        "net_bwd_time_dim": getattr(experiment_cfg, "net_bwd_time_dim", None),
        "leave_out_list": getattr(experiment_cfg, "leave_out_list", None),
    }

    return params


def load_test_datasets(experiment_path: str):
    """
    Charge les datasets de test depuis le dossier d'expérience.

    Args:
        experiment_path: Chemin vers le dossier d'expérience

    Returns:
        List[TimedDataset]: Liste des datasets de test chargés
    """
    import torch
    from datasets.datasets import TimedDataset

    exp_path = Path(experiment_path)
    datasets_test_dir = exp_path / "datasets_test"

    if not datasets_test_dir.exists():
        raise FileNotFoundError(
            f"Le dossier datasets_test/ n'existe pas dans {exp_path}"
        )

    # Chercher tous les fichiers .pt dans le dossier datasets_test
    dataset_files = list(datasets_test_dir.glob("test_dataset_time_*.pt"))

    if not dataset_files:
        raise FileNotFoundError(
            f"Aucun fichier de dataset de test trouvé dans {datasets_test_dir}"
        )

    # Charger et créer les TimedDataset
    test_datasets = []
    for file_path in sorted(dataset_files):  # Trier pour un ordre consistant
        dataset_dict = torch.load(file_path, map_location="cpu", weights_only=True)

        # Recréer le TimedDataset
        timed_dataset = TimedDataset(
            data=dataset_dict["data"], time=dataset_dict["time"]
        )
        test_datasets.append(timed_dataset)
        print(f"Loaded test dataset for time {dataset_dict['time']} from {file_path}")

    return test_datasets
