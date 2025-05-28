#bridge/trainer_dsbm.py

import torch
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt






""" In this file we implement the main class for the training of the Schrodinger Bridge framework with N constraints
"""


# class N_BRIDGES:

#     def __init__(self, )