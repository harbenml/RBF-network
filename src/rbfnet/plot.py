def plot_rbf(x, phi):
    for i in range(phi.shape[1]):
        plt.plot(x, phi[:, i])
