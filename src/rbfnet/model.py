from typing import List, Tuple

import numpy as np


class Model:
    def __init__(self):
        self.parameters: List[float]

    def predict(self, X: np.ndarray) -> np.ndarray:
        w = self.parameters
        y = X @ w
        return y
