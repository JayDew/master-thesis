import numpy as np
import matplotlib.pyplot as plt

from common import get_params, get_data, f

n, m = get_params()
Q, q, A, b = get_data()

tau = 1

v_init = np.random.normal(0, 1, n)
lambd_init = np.random.rand(n)

v = v_init
lambd = lambd_init
primal_res = []
dual_res = []
iters = np.arange(0, 500)

cached_inv = np.linalg.inv((Q + tau * np.matmul(A.T, A)))

for k in iters:
    u = np.matmul(cached_inv, np.dot(A.T, lambd + tau * v) - q)  # u-update
    v_prev = v
    v = np.minimum(np.dot(A, u) - (lambd / tau), b)  # v-update
    lambd = lambd + tau * (v - np.dot(A, u))  # lambda-update

    primal_res.append(np.linalg.norm(v - np.dot(A, u), 2))
    dual_res.append(np.linalg.norm(-tau * np.matmul(A.T, v - v_prev), 2))

print("Optimal value:", f(u))

plt.figure(figsize=(8, 6))
plt.plot(primal_res, label="Primal residual")
plt.plot(dual_res, label="Dual residual")
plt.yscale("log")
plt.xlabel("Iterations")
plt.ylabel("Residual Error")
plt.legend()
plt.grid()
plt.show()
