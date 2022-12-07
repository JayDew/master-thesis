import numpy as np
from matplotlib import pyplot as plt

from common import get_params

n, m, N = get_params()

cvxopt = np.load(f'Temp/cvxopt_{n}_{m}_{N}.npy')
# qophe = np.load(f'Temp/QOPHE_{n}_{m}_{N}.npy')
qophe_enc = np.load(f'Temp/QOPHE_encrypted_{n}_{m}_{N}.npy')
# alexandru = np.load(f'Temp/Alexandru_{n}_{m}_{N}.npy')

plt.plot(cvxopt, label="optimal solution")
# plt.plot(qophe, label="unencrypted QOPHE")
plt.plot(qophe_enc, label="Shoukry et al.")
# plt.plot(alexandru, label="Alexandru et al.")
plt.legend(loc="lower right")

plt.show()
