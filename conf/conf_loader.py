# conf/conf_loader.py
#
import importlib
import inspect
import shutil
from pathlib import Path


def load_config(config_name: str):
    """
    Load DistributionConfig and ExperimentConfig from conf.conf_classes.<config_name>,
    attach the distribution to the experiment config, copy the config file into
    the experiment's conf/ folder, and return both the ExperimentConfig and DistributionConfig.
    """
    module_path = f"conf.conf_classes.{config_name}"
    module = importlib.import_module(module_path)

    DistributionConfigCls = getattr(module, "DistributionConfig")
    ExperimentConfigCls = getattr(module, "ExperimentConfig")

    distribution_cfg = DistributionConfigCls()
    experiment_cfg = ExperimentConfigCls()
    experiment_cfg.distributions = distribution_cfg

    exp_root = Path(experiment_cfg.experiment_dir)
    exp_path = exp_root / experiment_cfg.experiment_name
    conf_dir = exp_path / "conf"
    conf_dir.mkdir(parents=True, exist_ok=True)

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
            lines.append(f"{name:20}: DistributionConfig with {len(distribution_cfg.distributions_train)} distributions")
        else:
            lines.append(f"{name:20}: {value}")
    lines.append("-" * 35)

    lines.append("Loaded Distribution Bridges:")
    lines.append("-" * 35)
    for dist in distribution_cfg.distributions_train:
        dist_type = type(dist).__name__

        # Extract all public attributes as parameters
        dist_params = {
            k: v for k, v in vars(dist).items()
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

    return experiment_cfg, distribution_cfg
