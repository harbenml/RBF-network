from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from example_process import polynomial
from rbf_kernel import calculate_radial_basis_function_matrix
from rbf_kernel import normalize_rbf


@dataclass
class RBFParameters:
    """Radial basis function parameters
    
    rbf_centers (np.ndarray): center coordinate of each rbf
    rbf_std (np.ndarray): standard deviations to construct the rbfs
    """

    centers: np.ndarray
    stds: np.array


@dataclass
class NonlinearParameters:
    """Holds the parameters of the radial basis functions (rbf) """

    rbfs: List[RBFParameters]
    smoothness: float


class Partitioning:
    """Class that holds the methods to calculate the neural network partitions  """

    @staticmethod
    def get_regression_matrix(
        input: np.ndarray, params: NonlinearParameters
    ) -> np.ndarray:
        num_inputs, num_rbfs = input.shape[0], len(params.rbfs)
        membership_fcns = np.zeros(num_inputs, num_rbfs)
        for i, rbf in enumerate(params.rbfs):
            membership_fcns[:, i] = calculate_radial_basis_function_matrix(
                input, rbf.centers, rbf.stds
            )


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

    N = 100
    number_of_neurons = 5
    smoothness = 1.5

    stds = np.ones(number_of_neurons) * smoothness / 10
    centers = np.linspace(0, 1, number_of_neurons)

    x = np.array([np.linspace(0, 1, N)])
    y = polynomial(x, sigma_noise=0.05)

    plt.plot(x.T, y.T, ".")
    plt.show()

