import numpy as np
import paillier

n = m = 20
N = 7
DEFAULT_KEYSIZE = 1024


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


fileH = "Data/H" + str(n) + "_" + str(m) + "_" + str(N) + ".txt"
H = np.loadtxt(fileH, delimiter=',')
assert is_pos_def(H) and check_symmetric(H), 'H must be a positive semi-definite matrix'

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
        for j in range(i_max, n * N):
            u_copy[j] = 0
        intermediate_values[int(i / n)] = f(u_copy)
    # save to file
    np.save(f'Temp/{filename}_{n}_{m}_{N}_optimum_solution.npy', u_opt)
    np.save(f'Temp/{filename}_{n}_{m}_{N}.npy', intermediate_values)


def is_solution_feasible(x):
    """
    Check that the constraints are met.

    Condition E*x < e should hold.
    """
    temp = np.matmul(E, x) < e
    print("Number of constraints violations:", np.count_nonzero(temp == False))
    return False if False in temp else True
