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
