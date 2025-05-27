# utils/metrics

from torch import Tensor



def get_classic_metrics(data: Tensor)-> float:

    # data Tensor with all the data from test set
    # Return mean, cov, std of the dataset

    return data.mean(dim = 1), data.std(dim=1, unbiased=False)



