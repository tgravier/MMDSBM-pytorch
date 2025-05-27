#models/networks

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import List

""" Here we design all the networks for the base case and the 2 marginals case"""

class MLP(nn.Module):
    def __init__(
            self,
            input_dim:int,
            layers_widths:List = [100,100,2],
            activation_fn = F.tanh
 
    )-> None:
        super(MLP, self).__init__()


        layers = []
        prev_width = input_dim
        for layer_width in layers_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            prev_width = layer_width
        
        self.input_dim = input_dim
        self.layers_widths = layers_widths
        self.layers = nn.ModuleList(layers)
        self.activation_fn = activation_fn
    
    def forward(self, x) -> Tensor:
        for i, layer in enumerate(self.layers[:-1]):

            x = self.activation_fn(layer(x))
        
        x = self.layers[-1](x)

        return x

class ScoreNetwork(nn.Module):

    def __init__(
            self,
            input_dim:int,
            layers_widths: List[int],
            activation_fn = F.tanh
    )-> None:
        
        super().__init__()
        self.net = MLP(
            input_dim,
            layers_widths = layers_widths,
            activation_fn= F.tanh
        )
    
    def forward(
            self,
            x_input:Tensor,
            t: Tensor
    )-> Tensor:
        
        inputs = torch.cat([x_input, t], dim = 1)
        return self.net(inputs)
    
    