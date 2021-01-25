import numpy as np

# estimate parameters
w_hat = np.linalg.pinv(X) @ y

