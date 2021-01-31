import numpy as np


def polynomial(x: np.ndarray, sigma_noise: float = 0) -> np.ndarray:
    y = 5 + x + 2 * x ** 2 - 3 * x ** 3 + sigma_noise * np.random.randn(max(x.shape), 1)
    return y
