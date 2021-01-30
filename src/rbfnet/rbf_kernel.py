import numpy as np


def calculate_radial_basis_function_matrix(
    x: np.ndarray, centers: np.ndarray, stds: np.ndarray
) -> np.ndarray:
    """Calculates the radial basis functions (RBFs) given the parameters for each
       neuron.

    Args:
        x (np.ndarray): input samples with dim [#samples, #dimensions]
        centers (np.ndarray): the centers of each Gaussian function  [#neurons, #dimensions]
        stds (np.ndarray): the standard deviations of each Gaussian function  [#neurons, #dimensions]

    Returns:
        rbf_matrix (np.ndarray): 2D numpy array containing the RBF values [#samples, #neurons]
    """
    rbf_matrix = np.exp(-0.5 * np.square((x - centers) / stds))
    return rbf_matrix


def normalize_rbf(rbf_matrix):
    """Normalizes each rbf neuron with the sum of all neuron values to from a
       `partition of unity`. 
       
    Args:
        rbf_matrix (np.ndarray): 2D numpy array containing the RBF values [#samples, #neurons]

    Returns:
        rbf_matrix_norm (np.ndarray): the normalized RBF values [#samples, #neurons]
    """
    column_sum = np.sum(rbf_matrix, axis=-1)
    rbf_matrix_norm = np.zeros(rbf_matrix.shape)
    for i in range(rbf_matrix.shape[1]):
        rbf_matrix_norm[:, i] = (
            rbf_matrix[:, i] / column_sum
        )  # TODO: is it possible to vectorize this?
    return rbf_matrix_norm

