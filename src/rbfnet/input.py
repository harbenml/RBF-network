import numpy as np

from typing import Tuple


def normalize_data(data: np.ndarray) -> Tuple[float, float, float]:
    mean_values = np.mean(data, axis=0)
    std_values = np.std(data, axis=0)
    data_norm = (data - mean_values) / std_values
    return data_norm, mean_values, std_values
