# runners/run_dsbm.py


# TODO Do this run DSBM and create the full architecture

import torch
import hydra
import os

from bridge.trainer_dsb import IMF_DSBM
from bridge.runners.config_getters import get_datasets, get_valid_test_datasets
from accelerate import Accelerator


def run(args):
    accelerator = Accelerator(cpu=args.device == "cpu", split_batches=True)
    accelerator.print("Directory: " + os.getcwd())

    init_ds, final_ds, mean_final, var_final = get_datasets(args)
    valid_ds, test_ds = get_valid_test_datasets(args)

    final_cond_model = None
    imf = IMF_DSBM(
        init_ds,
        final_ds,
        mean_final,
        var_final,
        args,
        accelerator=accelerator,
        final_cond_model=final_cond_model,
        valid_ds=valid_ds,
        test_ds=test_ds,
    )
    accelerator.print(accelerator.state)
    accelerator.print(imf.net["b"])
    accelerator.print(
        "Number of parameters:",
        sum(p.numel() for p in ipf.net["b"].parameters() if p.requires_grad),
    )
    ipf.train()
