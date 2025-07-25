#bridge/runners/ema.py
import torch
import copy
from torch import nn

class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay=0.999):
        super().__init__()
        self.ema_model = copy.deepcopy(model)
        self.decay = decay
        self.ema_model.eval()

        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def update(self, model: nn.Module):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data * (1. - self.decay))

    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.ema_model.to(*args, **kwargs)
        return self

    def state_dict(self, *args, **kwargs):
        return self.ema_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.ema_model.load_state_dict(state_dict, *args, **kwargs)

    def parameters(self, recurse: bool = True):
        return self.ema_model.parameters(recurse)

    def named_parameters(self, recurse: bool = True):
        return self.ema_model.named_parameters(recurse)

    def eval(self):
        self.ema_model.eval()
        return self

    def train(self, mode=True):
        self.ema_model.train(mode)
        return self
