from typing import Tuple

import numpy as np


def normalize_data(data: np.ndarray) -> Tuple[float, float, float]:
    mean_values = np.mean(data, axis=0)
    std_values = np.std(data, axis=0)
    data_norm = (data - mean_values) / std_values
    return data_norm, mean_values, std_values


def calculate_radial_basis_function_matrix(x, centers, stds):
    # x: (#samples, #dimensions)
    # centers: (#neurons, #dimensions)
    # stds: (#neurons, #dimensions)
    return np.exp(-0.5 * np.square((x - centers) / stds))


def normalize_rbf(rbf_matrix):
    # rbf_matrix: (#samples, #neurons)
    column_sum = np.sum(rbf_matrix, axis=-1)
    rbf_matrix_norm = np.zeros(rbf_matrix.shape)
    for i in range(rbf_matrix.shape[1]):
        rbf_matrix_norm[:, i] = rbf_matrix[:, i] / column_sum
    return rbf_matrix_norm


def plot_rbf(x, phi):
    for i in range(phi.shape[1]):
        plt.plot(x, phi[:, i])


def example_process(x, sigma_noise=0):
    y = 5 + x + 2 * x ** 2 - 3 * x ** 3 + sigma_noise * np.random.randn(1, max(x.shape))
    return y


def recursive_least_squares(x, y, theta_prev, P_prev, forgetting_factor=1):
    y_hat = x.T @ theta_prev
    error = y - y_hat
    correction = P_prev @ x / (x.T @ P_prev @ x + forgetting_factor)
    theta = theta_prev + correction @ error
    P = (np.eye(P_prev.shape[0]) - correction @ x.T) @ P_prev / forgetting_factor
    return theta, P
