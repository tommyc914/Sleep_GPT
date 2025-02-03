import numpy as np
from numpy import random

def gen_xy(
    model,
    iv_corr,
    n_obs):
    n_ivs = 10
    mean = np.zeros((n_ivs,))
    cov = np.block([[(iv_corr * np.ones((n_ivs - 3, n_ivs - 3)) + 
         (1 - iv_corr) * np.eye(n_ivs - 3)), np.zeros((7, 3))],
                    [np.zeros((3, 7)), np.eye(3)]])
    x = random.multivariate_normal(
      mean = mean, 
      cov = cov, 
      size = n_obs)
    if model == "linear":
        coef = np.array([.1, .2, .3, .4]).reshape(4, -1)
        cov_signal = cov[0:4, 0:4]
        error_var = 1 - (coef.T @ cov_signal @ coef).item()
        x_signal = x[:,0:4]
    else:
        coef = np.array([.3, .3, .3, .4]).reshape(4, -1)
        sd_quad = np.sqrt(2)
        sd_prod = np.sqrt(1 + iv_corr**2)
        a = (2 * (iv_corr**2)) / (sd_quad * sd_quad)
        b = (2 * (iv_corr**2)) / (sd_quad * sd_prod)
        cov_signal = np.array(
            [[ 1.  ,  0.  , 0.  ,  0.  ],
             [ 0.  ,  1.  ,  a,  b],
             [0.  ,  a,  1.  ,  b],
             [ 0.  ,  b,  b,  1.  ]])
        error_var = 1 - (coef.T @ cov_signal @ coef).item()
        x_signal = np.concatenate(
            (x[:,0:1], 
             (x[:,0:1]**2)  / sd_quad, 
             (x[:,1:2]**2) / sd_quad,
             (x[:,2:3] * x[:,3:4]) / sd_prod), 
            axis = 1)
    error = random.normal(
      loc = 0.0, 
      scale = np.sqrt(error_var), 
      size = (n_obs, ))
    y = (x_signal @ coef).reshape(-1,) + error
    r2 = 1 - error_var
    return x, y, r2


