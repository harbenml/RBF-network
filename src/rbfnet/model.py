from typing import List, Tuple

import numpy as np


class Model:
    def __init__(self):
        self.parameters: List[float]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # least squares estimation
        w_hat = np.linalg.pinv(X) @ y
        self.parameters = w_hat

    def predict(self, X: np.ndarray) -> np.ndarray:
        w = self.parameters
        y = X @ w
        return y
