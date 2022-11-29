import numpy as np
import paillier

n = m = 5
N = 7
DEFAULT_KEYSIZE = 1024

fileH = "Data/H" + str(n) + "_" + str(m) + "_" + str(N) + ".txt"
H = np.loadtxt(fileH, delimiter=',')
# assert is_pos_def(H), "Check that H is positive definite"

fileF = "Data/F" + str(n) + "_" + str(m) + "_" + str(N) + ".txt"
F = np.loadtxt(fileF, delimiter=',')

filex0 = "Data/x0" + str(n) + "_" + str(m) + "_" + str(N) + ".txt"
x0 = np.loadtxt(filex0, delimiter=',')

r = (2 * np.matmul(x0, F))
# Add constraints: E*U < e
E = - np.eye(n * N, m * N)
e = (np.ones(n * N) * 10 ** 6)  # basically remove all constraints

filepub = "Keys/pubkey" + str(DEFAULT_KEYSIZE) + ".txt"
with open(filepub, 'r') as fin:
    data = [line.split() for line in fin]
Np = int(data[0][0])
pubkey = paillier.PaillierPublicKey(n=Np)

filepriv = "Keys/privkey" + str(DEFAULT_KEYSIZE) + ".txt"
with open(filepriv, 'r') as fin:
    data = [line.split() for line in fin]
p = int(data[0][0])
q = int(data[1][0])
privkey = paillier.PaillierPrivateKey(pubkey, p, q)


def f(u):
    """
    Objective function that we want to minimize.
    Let r = 2*x0*F

    minimize (1/2)*u'*H*u + r'*u
    """
    return 0.5 * np.matmul(u.T, np.matmul(H, u)) + np.matmul(r.T, u)


def get_params():
    return n, m, N


def get_data():
    return H, r, E, e


def get_keys():
    return pubkey, privkey


def save_intermediate_variables(u_opt, filename):
    """
    Saves intermediate values in npy file.
    """
    intermediate_values = np.zeros(N)
    for i in range(0, n * N - 1, +n):
        i_max = i + n
        u_copy = u_opt.copy()
        for j in range(0, i):
            u_copy[j] = 0
        for j in range(i_max, n * N - 1):
            u_copy[j] = 0
        intermediate_values[int(i / n)] = f(u_copy)

    np.save(f'Temp/{filename}.npy', intermediate_values)
