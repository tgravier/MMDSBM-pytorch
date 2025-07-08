import torch
from torch import nn


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth = bandwidth
        self.register_buffer(
            "bandwidth_multipliers",
            mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        )

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        base_bw = self.get_bandwidth(L2_distances)

        # Assure que tout est sur le bon device
        device = L2_distances.device
        bandwidth_multipliers = self.bandwidth_multipliers.to(device)

        scaled_kernels = torch.exp(
            -L2_distances[None, ...] / (base_bw * bandwidth_multipliers[:, None, None])
        )
        return scaled_kernels.sum(dim=0)


class MMDLoss(nn.Module):
    def __init__(self, kernel=None):
        super().__init__()
        self.kernel = kernel if kernel is not None else RBF()

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()

        return XX - 2 * XY + YY
