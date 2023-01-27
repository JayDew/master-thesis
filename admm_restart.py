import numpy as np
import matplotlib.pyplot as plt

from common import get_params, get_data, f

n, m = get_params()
Q, q, A, b = get_data()

tau = 1

v_init = np.random.normal(0, 1, n)
lambd_init = np.random.rand(n)

v = v_init
v_hat = v_init
lambd = lambd_init
lambd_hat = lambd_init
alpha = 1
c = 0

primal_res_accr = []
dual_res_accr = []
iters = np.arange(0, 500)

cached_inv = np.linalg.inv((Q + tau * np.matmul(A.T, A)))

eta = 0.999
for k in iters:
    u = np.matmul(cached_inv, np.dot(A.T, lambd_hat + tau * v_hat) - q)

    v_prev = v
    v = np.minimum(np.dot(A, u) - (lambd_hat / tau), b)

    lambd_prev = lambd
    lambd = lambd_hat + tau * (v - np.dot(A, u))

    c_prev = c
    c = (np.linalg.norm(lambd - lambd_hat, 2) ** 2) / tau + tau * (np.linalg.norm(v - v_hat, 2) ** 2)

    alpha_prev = alpha
    if c < eta * c_prev:
        alpha = (1 + np.sqrt(1 + 4 * (alpha ** 2))) / 2

        v_hat = v + (alpha_prev - 1) * (v - v_prev) / alpha
        lambd_hat = lambd + (alpha_prev - 1) * (lambd - lambd_prev) / alpha
    else:
        alpha = 1
        v_hat = v_prev
        lambd_hat = lambd_prev
        c = c / eta

    primal_res_accr.append(np.linalg.norm(v - np.dot(A, u), 2))
    dual_residual = np.linalg.norm(-tau * np.matmul(A.T, v - v_prev), 2)
    if dual_residual == 0:
        dual_res_accr.append(dual_res_accr[-1])
    else:
        dual_res_accr.append(dual_residual)

print("Optimal value:", f(u))

plt.figure(figsize=(8, 6))
# plt.plot(iters, primal_res, label="Primal residual")
# plt.plot(iters, dual_res, label="Dual residual")
plt.plot(iters, primal_res_accr, label="Primal residual (restart)")
plt.plot(iters, dual_res_accr, label="Dual residual (restart)")
plt.yscale("log")
plt.xlabel("Iterations")
plt.ylabel("Residual Error")
plt.legend()
plt.grid()
plt.show()
