# models/loss.py

from torch import Tensor
import torch.nn.functional as F

""" In the file we have all the loss, for N marginal, 2-Marginal, toy_exemple, scale_example """


def dsbm_loss(
        model_output: Tensor,
        x_start: Tensor,
        x_final: Tensor,
        t: Tensor,
        direction: str,

) -> Tensor:
    
    """
    Derive the loss for the drift of the DSBM model

    direction : 'forward' or 'backward'

    """
    
    match direction:
        
        case "forward":

            target = (x_final - x_start) / (1.0 - t).view(-1, 1)

        case "backward":

            target = (x_start - x_final) / t.view(-1, 1)

        case _:

            raise ValueError(f"Unknown direction : {direction} , use 'forward' or 'backward' ")
    
    return F.mse_loss(model_output, target)
