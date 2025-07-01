# conf/conf_loader.py
#
import importlib
import inspect
import shutil
from pathlib import Path


import importlib
import shutil
import inspect
from pathlib import Path


def load_config(config_name: str):
    """
    Load DistributionConfig and ExperimentConfig from conf.conf_classes.<config_name>,
    attach the distribution to the experiment config, copy the config file into
    the experiment's conf/ folder, and return both the ExperimentConfig and DistributionConfig.
    """
    try:
        module_path = f"conf.conf_classes.{config_name}"
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(f"Could not import configuration module '{module_path}': {e}")

    try:
        DistributionConfigCls = getattr(module, "DistributionConfig")
        ExperimentConfigCls = getattr(module, "ExperimentConfig")
    except AttributeError as e:
        raise AttributeError(f"Configuration module '{module_path}' must define 'DistributionConfig' and 'ExperimentConfig': {e}")

    distribution_cfg = DistributionConfigCls()
    experiment_cfg = ExperimentConfigCls()

    if not hasattr(experiment_cfg, "experiment_dir") or not hasattr(experiment_cfg, "experiment_name"):
        raise AttributeError("ExperimentConfig must have 'experiment_dir' and 'experiment_name' attributes.")
    if not hasattr(experiment_cfg, "debug"):
        raise AttributeError("ExperimentConfig must have a 'debug' attribute.")

    experiment_cfg.distribution_cfg = distribution_cfg

    exp_root = Path(experiment_cfg.experiment_dir)
    exp_path = exp_root / experiment_cfg.experiment_name

    #  Check if the experiment already exists
    conf_dir = exp_path / "conf"

    experiments_debug = "experiments_debug"  # Or set this to the correct debug directory name/path

    if not experiment_cfg.debug:
        if str(exp_root) == experiments_debug:
            raise ValueError(
                f"Experiment directory '{experiment_cfg.experiment_dir}' is set to the debug directory while debug mode is off. "
                f"Please check your configuration."
            )
        if exp_path.exists():
            raise FileExistsError(
                f"An experiment named '{experiment_cfg.experiment_name}' already exists in '{experiment_cfg.experiment_dir}'. "
                f"Please choose a different experiment name or remove the existing folder."
            )
        try:
            conf_dir.mkdir(parents=True, exist_ok=False)  # Will raise an error if it already exists
        except FileExistsError:
            raise FileExistsError(f"The configuration directory '{conf_dir}' already exists.")
        except Exception as e:
            raise OSError(f"Failed to create configuration directory '{conf_dir}': {e}")
    else:
        if str(exp_root) != experiments_debug:
            raise ValueError(
                f"Debug mode is enabled but experiment directory '{experiment_cfg.experiment_dir}' is not set to the debug directory '{experiments_debug}'. "
                f"Please check your configuration."
            )
        # In debug mode, allow overwriting and create the conf_dir if needed
        try:
            conf_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OSError(f"Failed to create configuration directory '{conf_dir}': {e}")

    src_path = Path(module.__file__)
    dst_path = conf_dir / f"{config_name}.py"
    shutil.copy2(src_path, dst_path)

    # ───── Prepare formatted output
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

        # Extract all public attributes as parameters
        dist_params = {
            k: v
            for k, v in vars(dist).items()
            if not k.startswith("_") and not callable(v)
        }

        param_str = ", ".join(f"{k}={v}" for k, v in dist_params.items())
        lines.append(f"{dist_type:15} | {param_str}")
    lines.append("-" * 35)

    lines.append(f"Config file copied to: {dst_path.resolve()}")

    # Print to console
    print("\n" + "\n".join(lines) + "\n")

    # Save summary
    summary_path = conf_dir / f"{config_name}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    return experiment_cfg


def export_config_dict(config_classes):
    """
    Export a clean, wandb-compatible dictionary from the experiment and distribution configs.

    - Extracts simple attributes from experiment_cfg
    - Includes a summary of all training distributions from distribution_cfg.params
    """

    experiment_cfg, distribution_cfg = config_classes, config_classes.distribution_cfg
    config_dict = {}

    # Collect all non-callable, non-private attributes from experiment_cfg
    for name, value in vars(experiment_cfg).items():
        if name.startswith("_"):
            continue
        if name in {"distribution_cfg", "accelerator"}:
            continue
        if callable(value):
            continue
        config_dict[name] = value

    # Add a clean summary of the distributions using their .params dict
    config_dict["distributions"] = [
        {
            "type": type(dist).__name__,
            "time": dist.time,
            **dist.params,  # unpack all custom distribution parameters (mean, std, etc.)
        }
        for dist in distribution_cfg.distributions_train
    ]

    return config_dict
