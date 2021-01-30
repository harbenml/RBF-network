from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Tuple

import numpy as np

from rbf_kernel import calculate_radial_basis_function_matrix
from rbf_kernel import normalize_rbf


@dataclass
class RBFParameters:
    """Radial basis function parameters
    
    rbf_centers (np.ndarray): center coordinate of each rbf
    rbf_std (np.ndarray): standard deviations to construct the rbfs
    """

    center: np.ndarray
    std: np.array


@dataclass
class NonlinearParameters:
    """Holds the parameters of the radial basis functions (rbf) """

    rbfs: List[RBFParameters]
    smoothness: float


class Model:
    def __init__(self):
        self.linear_params: np.ndarray
        self.nonlin_params: NonlinearParameters

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
