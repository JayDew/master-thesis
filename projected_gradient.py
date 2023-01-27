import numpy as np
from common import get_params, get_data, f

n, m = get_params()
Q, c, A, b = get_data()

mu = np.random.normal(0, 1, n)
step_size = 0.05

primal_res = []
iters = np.arange(0, 100)

cached_inv = - (A @ np.linalg.inv(Q))

for k in iters:
    u = np.matmul(cached_inv, np.dot(A.T, mu) + c) - b  # gradient-update
    mu = np.maximum(np.zeros(n), mu + step_size * u)  # mu update

x_opt = - np.linalg.inv(Q) @ (A.T @ mu + c)
print("Optimal value:", f(x_opt))
