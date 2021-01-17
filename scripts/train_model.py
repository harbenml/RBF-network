import numpy as np

from example_process import polynomial


N = 10
x = np.array([np.linspace(0, 1, N)]).T
y = polynomial(x, sigma_noise=0.05)
