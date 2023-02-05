import numpy as np
from Pyfhel import Pyfhel
from scipy.optimize import linprog
from functools import reduce

from data import import_data

np.random.seed(42)

HE = Pyfhel()  # Creating empty Pyfhel object
ckks_params = {
    'scheme': 'CKKS',
    'n': 2 ** 14,  # Polynomial modulus degree
    'scale': 2 ** 30,  # All the encodings will use it for float->fixed point
    'qi_sizes': [60, 30, 30, 30, 60]  # Number of bits of each prime in the chain
}
HE.contextGen(**ckks_params)  # Generate context for bfv scheme
HE.keyGen()  # Key Generation: generates a pair of public/secret keys
HE.rotateKeyGen()


def encrypt_vector(x):
    return [HE.encrypt(y) for y in x]


def encrypt_matrix(x):
    return [[HE.encrypt(y) for y in z] for z in x]


def decrypt_vector(x):
    return [HE.decrypt(i)[0] for i in x]


def sum_encrypted_vectors(x, y):
    return [x[i] + y[i] for i in range(np.size(x))]


def diff_encrypted_vectors(x, y):
    return [x[i] - y[i] for i in range(len(x))]


def mul_sc_encrypted_vectors(x, y):  # x is encrypted, y is plaintext
    return [y[i] * x[i] for i in range(len(x))]


def dot_sc_encrypted_vectors(x, y):  # x is encrypted, y is encrypted
    return reduce(lambda x, y: x + y, mul_sc_encrypted_vectors(x, y))


def dot_m_encrypted_vectors(x, A):
    return [dot_sc_encrypted_vectors(x, vec) for vec in A]


def inv(m):
    a, b = m.shape
    i = np.eye(a, a)
    return np.linalg.lstsq(m, i)[0]


n = 5
c, A, b = import_data(n)

sol = linprog(c, A_eq=A, b_eq=b)
opt = sol['fun']
print('OPT:', opt, '---', sol['x'])

###################################

step_size = 0.01

P = np.eye(n) - A.T @ inv(A @ A.T) @ A
Q = A.T @ np.linalg.inv(A @ A.T) @ b

mu_enc = encrypt_vector(np.zeros(n))
enc_c = encrypt_vector(c)
enc_c_minus = encrypt_vector((step_size * (-c)))
enc_b = encrypt_vector(b)
enc_P = encrypt_matrix(P)
enc_Q = encrypt_vector(Q)


def _proj(x):
    return sum_encrypted_vectors(dot_m_encrypted_vectors(x, enc_P), enc_Q)


def objective(x):
    return c @ x


def gradient(x):
    return enc_c_minus


K = 200
for k in range(K):
    # server computes projected gradient descent
    u_enc = _proj(sum_encrypted_vectors(mu_enc, gradient(mu_enc)))
    # server sends back result to client for comparison
    u_dec = decrypt_vector(u_enc)
    # clients locally ensures that values are positive
    mu_new = np.maximum(np.zeros(n), u_dec)
    # client encrypts result and sends back to server
    mu_enc = encrypt_vector(mu_new)

    # the client receives the final result
    mu_dec = decrypt_vector(mu_enc)
    opt_privacy = objective(mu_dec)
    print(k, 'OPT:', f'{opt_privacy:.3f}', '---', mu_dec)

eplison = 1.e-1
assert np.isclose(opt_privacy, opt, eplison)
