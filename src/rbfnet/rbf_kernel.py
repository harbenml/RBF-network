from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from model import RBFParameters


def calculate_rbf(x: np.ndarray, params: RBFParameters) -> np.ndarray:
    """Calculates the radial basis functions (RBFs) given the parameters for each
       neuron.

    Args:
        x (np.ndarray): input samples with dim [#samples, #dimensions]
        centers (np.ndarray): the centers of each Gaussian function  [#neurons, #dimensions]
        stds (np.ndarray): the standard deviations of each Gaussian function  [#neurons, #dimensions]

    Returns:
        rbf_matrix (np.ndarray): 2D numpy array containing the RBF values [#samples, #neurons]
    """
    centers = params.centers
    stds = params.stds * params.smoothness / 20
    rbf_matrix = np.exp(-0.5 * np.square((x - centers) / stds))
    return rbf_matrix


def normalize_rbf(rbf_array):
    """Normalizes each rbf neuron with the sum of all neuron values to from a
       `partition of unity`. 
       
    Args:
        rbf_array (np.ndarray): 2D numpy array containing the RBF values [#samples, #neurons]

    Returns:
        rbf_array_norm (np.ndarray): the normalized RBF values [#samples, #neurons]
    """
    column_sum = np.c_[rbf_array.sum(axis=1)]
    rbf_array_norm = rbf_array / column_sum
    return rbf_array_norm

