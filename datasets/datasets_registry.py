# datasets/dataset_registry.py

# Some function to check if the dataset is valid

ARTIFICIAL_DATASETS = ["gaussian", "spiral", "moon"]
REAL_DATASETS = ["mnist", "cifar10"]

def is_valid_distribution(name: str, type_: str) -> bool:
    if type_ == "artificial":
        return name in ARTIFICIAL_DATASETS
    elif type_ == "real":
        return name in REAL_DATASETS
    return False

def list_distributions(type_: str) -> list:
    if type_ == "artificial":
        return ARTIFICIAL_DATASETS
    elif type_ == "real":
        return REAL_DATASETS
    return []
