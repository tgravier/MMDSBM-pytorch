from typing import Dict, Tuple
from bridge.core_dsbm import IMF_DSBM




""" File and class which there is the abstract function to begin the training"""


class Bridges_Trainer:

    def __init__(
            self,
            args,
            batch_size:int,
            scheduler,
            num_epoch,
            lr : float,
            optimisers,
            n_distribution : int,

    ):
        
        raise NotImplementedError
    

    def train(self, method):

        match method:

            case "direct":

                self._direct_train()
            
            case "bouncing":

                self._bouncing_direct_train()
            
    def _direct_train(
            self
    ):
            


        # do all the forward and after all the backward in the good order (1 outer)
        # TODO begin with this technic

        raise NotImplementedError
    
    def _bouncing_direct_train(
            self
    ):

        # do forward-backward and go at the next bridge (1 outer)
        # TODO second technic to developp

        raise NotImplementedError
    
