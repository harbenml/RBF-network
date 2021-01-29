from dataclasses import dataclass, field
from typing import List, NamedTuple, Tuple

import numpy as np


@dataclass
class RBF:
    """Radial basis function parameters """

    center: np.ndarray
    std: np.array


@dataclass
class Partitioning:
    """Holds the parameters of the radial basis functions (rbf)
    
    rbf_centers (np.ndarray): center coordinate of each rbf
    rbf_std (np.ndarray): standard deviations to construct the rbfs
    """

    rbfs: List[RBF]


# class Partitioning(NamedTuple):
#     """Holds the parameters of the radial basis functions (rbf)

#     rbf_centers (np.ndarray): center coordinate of each rbf
#     rbf_std (np.ndarray): standard deviations to construct the rbfs
#     """

#     rbf_centers: np.ndarray
#     rbf_std: np.ndarray


class Model:
    def __init__(self):
        self.linear_params: np.ndarray
        self.nonlin_params: Partitioning

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # least squares estimation
        w_hat = np.linalg.pinv(X) @ y
        self.linear_params = w_hat

    def predict(self, X: np.ndarray) -> np.ndarray:
        w = self.linear_params
        y = X @ w
        return y


if __name__ == "__main__":
    parameters = Partitioning(0, 0)
    print(parameters)
