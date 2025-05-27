#datasets/datasets.py

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


from conf.conf import ConfigLoader


"""

This file contains the Datasets class, which is used to instanciate from a config class a dataset from artificial/real unpaired data.


"""


## Test d'une config

config_path_str = "/projects/static2dynamic/Gravier/schron/conf/config_dataset_example.yaml"
config_path = os.path.abspath(config_path_str)
config_loader = ConfigLoader(config_path)

print(config_loader.get_config())

### TODO a enlever après DEBUG

### create a fonction to see if the artificial/real data exist

# dataset.py


