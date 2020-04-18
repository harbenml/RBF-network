# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

import matplotlib.pyplot as plt
import numpy as np

# +
N = 100
sigma_noise = 0
x = np.linspace(0, 1, num=N)

np.random.seed(23)
y = 1/(0.1 + x) + sigma_noise*np.random.randn(1, N); y = y.T
# -

plt.plot(x, y);
plt.grid()


# **Functions to implement:**
# 1. normalize data
# 2. calculate radial basis functions
# 3. build regression matrix
# 4. estimate parameters using weighted least squares

def normalize_data(data):
    mean_values = np.mean(data, axis=0)
    std_values = np.std(data, axis=0)
    data_norm = (data-mean_values)/std_values
    return data_norm, mean_values, std_values


def calculate_radial_basis_function_matrix(x, centers, stds):
    # x: (#samples, #dimensions)
    # centers: (#neurons, #dimensions)
    # stds: (#neurons, #dimensions)
    return np.exp( -0.5*np.square((x - centers) / stds) )


def normalize_rbf(rbf_matrix):
    # rbf_matrix: (#samples, #neurons)
    column_sum = np.sum(rbf_matrix, axis=-1)
    rbf_matrix_norm = np.zeros(rbf_matrix.shape)
    for i in range(rbf_matrix.shape[1]):
        rbf_matrix_norm[:,i] = rbf_matrix[:,i]/column_sum
    return rbf_matrix_norm


def plot_rbf(x, phi):
    for i in range(phi.shape[1]):
        plt.plot(x, phi[:,i]);


def example_process(x, sigma_noise=0):
    y = 5 + x + 2*x**2 - 3*x**3 + sigma_noise*np.random.randn(1, max(x.shape))
    return y


data_norm, mean_values, std_values = normalize_data(np.c_[x, 2*x])

np.std(np.c_[x, 2*x], axis=0)

np.std(data_norm, axis=0)

# +
N = 100
number_of_neurons = 5
smoothness = 1.5

stds = np.ones(number_of_neurons)*smoothness/10
centers = np.linspace(0, 1, number_of_neurons)
# centers = centers[1:-1]


x = np.array([np.linspace(0, 1, N)])
y = example_process(x, sigma_noise=0.05)
# y = 5 + x + 2*x**2 - 3*x**3 + 0.05*np.random.randn(1, N)
# y = 1/(0.1 + x)

x = x.T
y = y.T
# -

plt.plot(x,y,'.');

x.shape

phi = calculate_radial_basis_function_matrix(x, centers, stds)

phi_norm = normalize_rbf(phi)

plot_rbf(x, phi)

plot_rbf(x, phi_norm)

# +
X = np.c_[np.ones((N, 1)), phi_norm]
# X = phi_norm

# estimate parameters
w_hat = np.linalg.pinv(X)@y

# calculate model output
y_hat = X@w_hat

# NRMSE
np.sqrt(np.mean( (y_hat - y)**2 ))
# -

plt.plot(x, y_hat, 'r');
plt.plot(x, y, 'b.');

plot_rbf(x, phi_norm*w_hat[1:].T+w_hat[0])
plt.plot(x, y_hat, 'k');

w_hat

w_hat.shape

X.shape

y.shape

X.shape

phi.shape


# +
def recursive_least_squares(x, y, theta_prev, P_prev, forgetting_factor=1):
    y_hat = x.T @ theta_prev
    error = y - y_hat
    correction = P_prev@x/(x.T@P_prev@x+forgetting_factor)
    theta = theta_prev + correction@error
    P = (np.eye(P_prev.shape[0])-correction@x.T)@P_prev/forgetting_factor
    return theta, P

# function [theta_new, P_new] = rls(x, y, theta_old, P_old, lambda)

# y_hat = x'*theta_old;                               % model output with old parameter vector
# e = y - y_hat;                                      % error
# gamma = P_old*x/(x'*P_old*x+lambda);                % update for gamma
# theta_new = theta_old + gamma*e;                    % update for parameter vector
# P_new = (eye(size(P_old))-gamma*x')*P_old/lambda;   % update for covariance matrix



# +
P = np.eye(number_of_neurons+1)
forgetting_factor = 0.9
theta = np.zeros((6,1))
x_sim_plot = np.empty(0)
y_sim_plot = np.empty(0)

import time
from IPython import display

for i in range(30):
    x_sim = np.random.rand(1)
    
    y_sim = example_process(x_sim, sigma_noise=0)

#     set_trace()
    a = np.append(a, np.random.rand(1))
    x_sim_plot = np.append(x_sim_plot, x_sim)
    y_sim_plot = np.append(y_sim_plot, y_sim)
    
    phi_rls = calculate_radial_basis_function_matrix(x_sim, centers, stds)
    phi_rls_norm = normalize_rbf(np.reshape(phi_rls, (1,number_of_neurons)))
    regressor = np.c_[1, phi_rls_norm]

    theta, P = recursive_least_squares(regressor.T, y_sim, theta, P, forgetting_factor)
    

    plt.cla()
    plt.plot(x, X@theta);
    plt.plot(x_sim_plot, y_sim_plot, '.')
    plt.vlines(x_sim, 4, 6)
        
    display.clear_output(wait=True)
    display.display(plt.gcf())
    time.sleep(0.1)

plt.close()
#     print(theta)

# +
import matplotlib.pylab as plt
import pandas as pd
import numpy as np

# %matplotlib inline

i = pd.date_range('2013-1-1',periods=100,freq='s')

while True:
    try:


    except KeyboardInterrupt:
        break
# -

np.eye(P.shape[0])

from IPython.core.debugger import set_trace
set_trace()

a = np.empty(0)
a

a = np.append(a, np.random.rand(1))
print(a)


