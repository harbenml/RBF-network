def example_process(x, sigma_noise=0):
    y = 5 + x + 2 * x ** 2 - 3 * x ** 3 + sigma_noise * np.random.randn(1, max(x.shape))
    return y
