import sys
import torch
import numpy as np
import torch.nn as nn
from ap2 import approximate_power_of_two
from torch.nn.parameter import Parameter


def shift_based_batchnorm(X: torch.tensor) -> torch.tensor:
    with torch.no_grad():
        if len(X.shape) == 2:
            gamma = nn.BatchNorm1d(X.shape[1])
            gamma = Parameter(gamma(X))
        elif len(X.shape) == 4:
            gamma = nn.BatchNorm2d(X.shape[1])
            gamma = Parameter(gamma(X))
        elif len(X.shape) == 5:
            gamma = nn.BatchNorm3d(X.shape[1])
            gamma = Parameter(gamma(X))
        else:
            gamma = nn.SyncBatchNorm(X.shape[1])
            gamma = Parameter(gamma(X))

        mu = torch.mean(X)
        C = (X - mu)

        # bit_shift
        ap2_C = (approximate_power_of_two(C).numpy()).astype(np.int64)
        C = (C.numpy()).astype(np.int64)
        var = np.mean(C >> ap2_C).astype(float)

        ap2_var = (
            approximate_power_of_two((1 / torch.sqrt(torch.tensor(var) + sys.float_info.epsilon))).numpy()).astype(
            np.int64)
        X_hat = C >> ap2_var

        ap2_gamma = (approximate_power_of_two(gamma).numpy()).astype(np.int64)
        out = ap2_gamma >> X_hat
        out = torch.from_numpy(out)
        cache = (X, X_hat, mu, var, gamma)

    return out, cache


if __name__ == '__main__':
    X1d = torch.randn(2, 2)

    a = shift_based_batchnorm(X1d)
    b = nn.BatchNorm1d(2)(X1d)

    print("before tensor : \n", X1d)
    print("\n SBN tensor : \n", a[0])
    print("\n Standard BN tensor : \n", b)
