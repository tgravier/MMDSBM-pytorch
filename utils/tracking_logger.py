#utils/tracking_logger.py

import wandb

class WandbLogger:
    def __init__(self, project: str, config: dict = None, run_name: str = None):
        self.run = wandb.init(project=project, config=config, name=run_name)

    def log(self, data: dict, step: int = None):
        if step is not None:
            wandb.log(data, step=step)
        else:
            wandb.log(data)

    def watch(self, model, log="all"):
        wandb.watch(model, log=log)

    def finish(self):
        wandb.finish()
