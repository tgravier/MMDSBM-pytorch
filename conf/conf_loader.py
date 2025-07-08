import importlib
import importlib.util
import inspect
import shutil
from pathlib import Path


def load_config(
    config_name: str = None, experiment_path: str = None, resume_train: bool = False
):
    """
    Load configuration either from conf.conf_classes.<config_name> (normal mode)
    or from a saved config file inside experiment_path/conf/ (resume mode).
    """
    if resume_train:
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
    distribution_cfg = DistributionConfigCls()
    experiment_cfg = ExperimentConfigCls()
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
                f"{name:20}: DistributionConfig with {len(distribution_cfg.distributions_train)} distributions"
            )
        else:
            lines.append(f"{name:20}: {value}")
    lines.append("-" * 35)

    lines.append("Loaded Distribution Bridges:")
    lines.append("-" * 35)
    for dist in distribution_cfg.distributions_train:
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
        for dist in distribution_cfg.distributions_train
    ]
    return config_dict
