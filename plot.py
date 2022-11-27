import numpy as np
from matplotlib import pyplot as plt

cvxopt = np.load('Temp/cvxopt.npy')
qophe = np.load('Temp/QOPHE.npy')
alexandru = np.load('Temp/Alexandru.npy')

plt.plot(cvxopt, label="optimal solution")
plt.plot(qophe, label="unencrypted QOPHE")
plt.plot(alexandru, label="alexandru et al.")
plt.legend(loc="lower right")

plt.show()
