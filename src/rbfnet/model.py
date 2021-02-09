from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from example_process import polynomial
from rbf_kernel import calculate_rbf
from rbf_kernel import normalize_rbf


@dataclass
class RBFParameters:
    """Holds the parameters of the radial basis functions (rbf) """

    centers: np.ndarray
    stds: np.array
    smoothness: float


class Partitioning:
    """Class that holds the methods to calculate the neural network partitions  """

    @staticmethod
    def get_regression_matrix(input: np.ndarray, params: RBFParameters) -> np.ndarray:
        num_samples, num_rbfs = input.shape[0], len(params.centers)
        membership_fcns = calculate_rbf(input, params.centers, params.stds)
        phi = normalize_rbf(membership_fcns)
        return phi


class Model:
    def __init__(self):
        self.linear_params: np.ndarray
        self.rbf_params: RBFParameters

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # least squares estimation
        phi = self.get_regression_matrix(X)
        w_hat = np.linalg.pinv(X) @ y
        self.linear_params = w_hat

    def predict(self, X: np.ndarray) -> np.ndarray:
        phi = self.get_regression_matrix(X)
        w = self.linear_params
        y = X @ w
        return y

    def get_regression_matrix(self, input: np.ndarray) -> np.ndarray:
        num_samples, num_rbfs = input.shape[0], len(self.rbf_params.centers)
        membership_fcns = calculate_rbf(input, self.rbf_params)
        phi = normalize_rbf(membership_fcns)
        return phi


if __name__ == "__main__":

    N = 100
    number_of_neurons = 5
    smoothness = 1

    stds = np.ones(number_of_neurons) * smoothness / 20
    centers = np.linspace(0, 1, number_of_neurons)

    x = np.array([np.linspace(0, 1, N)])
    y = polynomial(x, sigma_noise=0.05)

    x = x.T
    y = y.T

    rbf_params = RBFParameters(centers, stds, smoothness=1)
    print(rbf_params)

    mu = calculate_rbf(x, centers, stds)

    phi = Partitioning.get_regression_matrix(x, rbf_params)
    # print(phi.shape)

    # print(np.sum(phi, axis=0))

