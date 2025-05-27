# conf/conf.py

## Debug ##

if __name__ == "__main__":
    import sys
    import os

    print("Debug mode")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)


## Debug ##

from dataclasses import dataclass, field
from typing import Literal, Dict, Optional, List
import yaml
import os

from datasets.datasets_registry import is_valid_distribution


"""

Class to hold the configuration for the Schrodinger Bridge.
This includes the configuration for the model, training, and normalisation.

"""




@dataclass
class NormalisationConfig:

    """
    
    Configuration for normalisation of the data.

    """
    active: bool = False
    type: Literal["standart", "minmax", "none"] = "none"


@dataclass

class MarginalConfig:
    
    """
    
    Configuration for the marginal distribution.

    """
    
    distribution_name: str
    time: float
    type : Literal["artificial", "real"] = "artificial"
    path: Optional[str] = None
    params: Dict = field(default_factory=dict)

@dataclass
class BridgeConfig:
    
    """
    
    Configuration for the Schrodinger Bridge with N or 2 marginals.

    """
    nb_marginals: int
    sep_train_val : bool = False
    train_frac: float = 0.8
    normalisation: NormalisationConfig = NormalisationConfig()
    marginals : List[MarginalConfig] = field(default_factory=list)

    def validate(self):
        """
        
        Validate the configuration.
        
        """
        if len(self.marginals) != self.nb_marginals:
            raise ValueError(f"Number of marginals {len(self.marginals)} does not match the number of marginals in the config {self.nb_marginals}.")
        
        # Validate dims for artificals

        dims = set()
        for m in self.marginals:

            

            if not is_valid_distribution(m.distribution_name, m.type): # Check if the distribution name is valid and use the function is_valid_distribution in conf.py for testing
                raise ValueError(f"Invalid distribution_name '{m.distribution_name}' for type '{m.type}' at time {m.time}.")


            if m.type == "artificial":
                dim = m.params.get("dim")
                if dim:
                    dims.add(dim)
            elif m.type == "real":

                if not m.distribution_name:
                    raise ValueError(f"Distribution/Dataset name is required for real marginals.")
                
                
                
                if not m.path:
                    raise ValueError("Path is required for real marginals.")
                if not os.path.exists(m.path):
                    
                    raise ValueError(f"Path {m.path} does not exist.")
            if len(dims) > 1:
                raise ValueError(f"All artificial marginals must have the same dimension. Found {dims}.")
            
            # Validate for train_frac
            if self.sep_train_val:
                if self.train_frac <= 0 or self.train_frac >= 1:
                    raise ValueError(f"train_frac must be between 0 and 1. Found {self.train_frac}.")
            
            # Validate for time interval
            if m.time < 0:
                raise ValueError(f"Time must be positive. Found {m.time}.")
            
            times = list() # Validate if each marginal has a unique time
            
            for m in self.marginals:
                if m.type == "artificial":
                    # Vérifie que params contient 'dim'
                    if not m.params or "dim" not in m.params:
                        raise ValueError(f"'dim' is required in params for artificial marginal at time {m.time}.")

                    if not m.distribution_name:
                        raise ValueError(f"'distribution_name' is required for artificial marginal at time {m.time}.")

                # Validate if each there is a unique time for each marginal

                times.append(m.time)
                if len(times) != len(set(times)):
                    raise ValueError(f"All marginals must have a unique time. Found {times}.")

                

class ConfigLoader:
    
    """
    
    Class to load the configuration from a file.
    
    """
    def __init__(self, config_path: str):

        self.config_path = config_path # Path of the main config file
        self.bridge_config = self._load_config() # Function to load the config file, use by the class and not by the user

    
    def _load_config(self):

        with open(self.config_path, "r") as f:

            config_dict = yaml.safe_load(f)

        normalisation = NormalisationConfig(**config_dict["normalisation"]) # ** permit to unpack the dict as arguments in the function
        marginals = [MarginalConfig(**m) for m in config_dict["marginals"]]

        bridge_config = BridgeConfig(

            nb_marginals=config_dict["nb_marginals"],
            sep_train_val=config_dict["sep_train_val"],
            train_frac=config_dict["train_frac"],
            normalisation=normalisation,
            marginals=marginals
        )

        # Validate the config

        bridge_config.validate()

        return bridge_config
    
    def get_config(self) -> BridgeConfig:

        """
        
        Get the main configuration.
        
        """
        return self.bridge_config

    def get_marginals(self) -> List[MarginalConfig]:
        
        """
        
        Get the marginals configuration.
        
        """
        return self.bridge_config.marginals
    
    def get_normalisation_type(self) -> NormalisationConfig:

        """
        
        Get the normalisation configuration.
        
        """
        return self.bridge_config.normalisation.type

    def is_normalisation_active(self) -> bool:

        """
        
        Check if normalisation is active.
        
        """
        return self.bridge_config.normalisation.active
    
    def get_train_frac(self) -> float:
        
        """
        
        Get the train fraction.
        
        """
        return self.bridge_config.train_frac
    
# TODO mettre plus de get et plus de verification dans la classe



