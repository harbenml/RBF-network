import numpy as np


class Loss:
    """
    Base class that implements a loss function
    """

    def loss(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        raise NotImplementedError


class RMSE(Loss):
    """
    RMSE: root mean squared error
    """

    def loss(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        return np.sqrt(np.mean((actual - predicted) ** 2))

