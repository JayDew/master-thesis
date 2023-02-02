import numpy as np
from scipy.optimize import linprog
from gmpy2 import mpz
import paillier
import gmpy2

DEFAULT_KEYSIZE = 1024
DEFAULT_MSGSIZE = 32
DEFAULT_PRECISION = int(DEFAULT_MSGSIZE / 2)  # of fractional bits
DEFAULT_FACTOR = 80
DEFAULT_STATISTICAL = 40  # The multiplication by random numbers offers DEFAULT_STATISTICAL security
NETWORK_DELAY = 0  # 10 ms

np.random.seed(42)


def encrypt_vector(pubkey, x, coins=None):
    if (coins == None):
        return [pubkey.encrypt(y) for y in x]
    else:
        return [pubkey.encrypt(y, coins.pop()) for y in x]


def encrypt_matrix(pubkey, x, coins=None):
    if (coins == None):
        return [[pubkey.encrypt(y) for y in z] for z in x]
    else:
        return [[pubkey.encrypt(y, coins.pop()) for y in z] for z in x]


def decrypt_vector(privkey, x):
    return [privkey.decrypt(i) for i in x]


def sum_encrypted_vectors(x, y):
    return [x[i] + y[i] for i in range(np.size(x))]


def diff_encrypted_vectors(x, y):
    return [x[i] - y[i] for i in range(len(x))]


def mul_sc_encrypted_vectors(x, y):  # x is encrypted, y is plaintext
    return [y[i] * x[i] for i in range(len(x))]


def dot_sc_encrypted_vectors(x, y):  # x is encrypted, y is plaintext
    return sum(mul_sc_encrypted_vectors(x, y))


def dot_m_encrypted_vectors(x, A):
    return [dot_sc_encrypted_vectors(x, vec) for vec in A]


####### We take the convention that a number x < N/3 is positive, and that a number x > 2N/3 is negative.
####### The range N/3 < x < 2N/3 allows for overflow detection.

def fp(scalar, prec=DEFAULT_PRECISION + DEFAULT_FACTOR):
    if prec < 0:
        return gmpy2.t_div_2exp(mpz(scalar), -prec)
    else:
        return mpz(gmpy2.mul(scalar, 2 ** prec))


def fp_vector(vec, prec=DEFAULT_PRECISION + DEFAULT_FACTOR):
    if np.size(vec) > 1:
        return [fp(x, prec) for x in vec]
    else:
        return fp(vec, prec)


def fp_matrix(mat, prec=DEFAULT_PRECISION + DEFAULT_FACTOR):
    return [fp_vector(x, prec) for x in mat]


def retrieve_fp(scalar, prec=DEFAULT_PRECISION + DEFAULT_FACTOR):
    return scalar / (2 ** prec)


def retrieve_fp_vector(vec, prec=DEFAULT_PRECISION + DEFAULT_FACTOR):
    return [retrieve_fp(x, prec) for x in vec]


def retrieve_fp_matrix(mat, prec=DEFAULT_PRECISION + DEFAULT_FACTOR):
    return [retrieve_fp_vector(x, prec) for x in mat]


filepub = "Keys/pubkey" + str(DEFAULT_KEYSIZE) + ".txt"
with open(filepub, 'r') as fin:
    data = [line.split() for line in fin]
Np = int(data[0][0])
pubkey = paillier.PaillierPublicKey(n=Np)

filepriv = "Keys/privkey" + str(DEFAULT_KEYSIZE) + ".txt"
with open(filepriv, 'r') as fin:
    data = [line.split() for line in fin]
p = mpz(data[0][0])
q = mpz(data[1][0])
privkey = paillier.PaillierPrivateKey(pubkey, p, q)

###################################

c = np.asarray([-2., -2.])
A = np.asarray([[1., 1.], [1., -1.]])
b = np.asarray([3., 0.])

sol = linprog(c, A_eq=A, b_eq=b)
print('OPT:', sol['fun'], '---', sol['x'])

###################################3

scaling_factor = 1
A = scaling_factor * A
P = np.eye(2) - A.T @ np.linalg.inv(A @ A.T) @ A
Q = A.T @ np.linalg.inv(A @ A.T) @ b

n = 2
mu = encrypt_vector(pubkey, fp_vector(np.random.normal(0, 1, n)))
step_size = 0.005

enc_c = encrypt_vector(pubkey, fp_vector(c))
enc_c_minus = encrypt_vector(pubkey, fp_vector(step_size * (-c)))
enc_b = encrypt_vector(pubkey, fp_vector(b))
enc_P = encrypt_matrix(pubkey, fp_matrix(P))
enc_Q = encrypt_vector(pubkey, fp_vector(Q))


def _proj(x):
    return sum_encrypted_vectors(dot_m_encrypted_vectors(x, fp_matrix(P)), enc_Q)


def objective(x):
    return c @ x


def gradient(x):
    return enc_c_minus


K = 100
for k in range(K):
    # server computes projected gradient descent
    u_enc = _proj(sum_encrypted_vectors(mu, gradient(mu)))
    # server sends back result to client for comparison
    u_dec = retrieve_fp_vector(decrypt_vector(privkey, u_enc))
    # clients locally ensures that values are positive
    mu_new = np.maximum(np.zeros(n), u_dec)
    # client encrypts result and sends back to server
    mu = encrypt_vector(pubkey, fp_vector(mu_new))

mu_dec = retrieve_fp_vector(decrypt_vector(privkey, mu))
print('OPT:', objective(mu_dec), '---', mu_dec)

# TODO! after lunch break:
#  2. Check if it really necessary to divulge A
#  3. Check if A leaks any information about the system in the first place
