def recursive_least_squares(x, y, theta_prev, P_prev, forgetting_factor=1):
    y_hat = x.T @ theta_prev
    error = y - y_hat
    correction = P_prev @ x / (x.T @ P_prev @ x + forgetting_factor)
    theta = theta_prev + correction @ error
    P = (np.eye(P_prev.shape[0]) - correction @ x.T) @ P_prev / forgetting_factor
    return theta, P
