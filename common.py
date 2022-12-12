import numpy as np
import paillier

n = m = 3
DEFAULT_KEYSIZE = 1024


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


fileQ = "Data_2/Q" + str(n) + "_" + str(m) + ".txt"
Q = np.loadtxt(fileQ, delimiter=',')
assert is_pos_def(Q) and check_symmetric(Q), 'Q must be a positive semi-definite matrix'

fileA = "Data_2/A" + str(n) + "_" + str(m) + ".txt"
A = np.loadtxt(fileA, delimiter=',')

fileb = "Data_2/b" + str(n) + "_" + str(m) + ".txt"
b = np.loadtxt(fileb, delimiter=',')

filec = "Data_2/c" + str(n) + "_" + str(m) + ".txt"
c = np.loadtxt(filec, delimiter=',')

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
    return 0.5 * np.matmul(u.T, np.matmul(Q, u)) + np.matmul(c.T, u)


def get_params():
    return n, m


def get_data():
    return Q, c, A, b


def get_keys():
    return pubkey, privkey


def save_intermediate_variables(u_opt, filename):
    """
    Save results in npy file.
    """
    optimum_objective_value = f(u_opt)
    # save to file
    np.save(f'Temp/{filename}_{n}_{m}_optimum_solution.npy', u_opt)
    np.save(f'Temp/{filename}_{n}_{m}.npy', optimum_objective_value)


def is_solution_feasible(x):
    """
    Check that the constraints are met.

    Condition E*x < e should hold.
    """
    temp = np.matmul(A, x) < b.reshape(-1, 1)
    print("Number of constraints violations:", np.count_nonzero(temp == False))
    return False if False in temp else True
