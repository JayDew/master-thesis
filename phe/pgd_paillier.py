import time

import numpy as np
from gmpy2 import mpz
from phe import paillier
import gmpy2

DEFAULT_KEYSIZE = 1024
DEFAULT_MSGSIZE = 32
DEFAULT_PRECISION = int(DEFAULT_MSGSIZE / 2)  # of fractional bits
DEFAULT_FACTOR = 10

np.random.seed(420)


def encrypt_vector(pubkey, x):
    return [pubkey.encrypt(y) for y in x]


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
        return mpz(gmpy2.mul(float(scalar), 2 ** prec))


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


filepub = "../Keys/pubkey" + str(DEFAULT_KEYSIZE) + ".txt"
with open(filepub, 'r') as fin:
    data = [line.split() for line in fin]
Np = int(data[0][0])
pubkey = paillier.PaillierPublicKey(n=Np)

filepriv = "../Keys/privkey" + str(DEFAULT_KEYSIZE) + ".txt"
with open(filepriv, 'r') as fin:
    data = [line.split() for line in fin]
p = mpz(data[0][0])
q = mpz(data[1][0])
privkey = paillier.PaillierPrivateKey(pubkey, p, q)

from scipy.optimize import linprog
from util.graphGenerator import GraphGenerator


def inv(A):
    return np.linalg.pinv(A)


def get_b_vector(N, s, t):
    b = np.zeros(N)
    b[s] = 1
    b[t] = -1
    return b


experiments = [
    (5, [20]),
    # (8, [56]),
    # (10, [90]),
    # (16, [210]),
    # (20, [380])
]

for exp in experiments:
    n = exp[0]
    Es = exp[1]
    for E in Es:
        results = np.asarray([np.NAN] * 8)
        for i in range(10):  # repeat each experiment 10 times
            generator = GraphGenerator(N=n, E=E, seed=i)  # generate graph
            e, c, A = generator.generate_random_graph()
            # c = c / np.linalg.norm(c)  # normalize cost vector
            longest_shortest_path = generator.get_longest_path()
            s = longest_shortest_path[0]
            t = longest_shortest_path[-1]
            b = get_b_vector(n, s, t)
            #################################
            # Exact solution using plaintext
            sol = linprog(c, A_eq=A, b_eq=b)
            opt = sol['fun']
            # print('OPT:', opt, '---', sol['x'])
            ###################################

            step_size = 0.001
            enc_c_minus = encrypt_vector(pubkey, fp_vector(step_size * (-c)))
            P = np.eye(e) - A.T @ inv(A @ A.T) @ A
            Q = A.T @ inv(A @ A.T) @ b
            Q_enc = encrypt_vector(pubkey, fp_vector(fp_vector(Q)))
            x0 = np.ones(e) * 0.5
            x0_enc = encrypt_vector(pubkey, fp_vector(x0))
            x0_dec = x0

            def objective(x):
                return c @ x


            def _proj(x):
                return sum_encrypted_vectors(dot_m_encrypted_vectors(x, fp_matrix(P)), Q_enc)


            def gradient(x):
                return enc_c_minus


            correct_value_after = 0
            correct_found = False
            # start measuring execution time
            start_time = time.time()
            fucked_up = False

            k = 0
            while True:
                k = k + 1
                print(k)
                if fucked_up:
                    break
                # cloud performs the projection
                x0_enc_new = _proj(sum_encrypted_vectors(x0_enc, gradient(x0_enc)))
                # sends to the client for decryption
                x0_dec_new = np.asarray(list(map(lambda x: float(x), np.maximum(np.zeros(e), retrieve_fp_vector(retrieve_fp_vector(decrypt_vector(privkey, x0_enc_new)))))))
                # client locally performs max and sends ecrypted vector
                x0_enc_new = encrypt_vector(pubkey, fp_vector(x0_dec_new))

                # correctness without convergence
                if np.allclose(np.rint(x0_dec_new), sol['x']) and not correct_found:
                    correct_found = True
                    correct_value_after = k

                if np.allclose(x0_dec, x0_dec_new):  # convergence
                    if not (np.isclose(objective(x0_dec_new), objective(sol['x']), rtol=1.e-2) or np.allclose(np.rint(x0_dec_new), sol['x'])):  # correctness
                        results = np.vstack((results, np.asarray([n, e, correct_value_after, k + 1, time.time() - start_time, 1, 0, (objective(x0_dec_new) - objective(sol["x"]))])))
                        fucked_up = True
                        continue
                    print(f'convergence after {k + 1} iterations')
                    results = np.vstack((results, np.asarray([n, e, correct_value_after, k + 1, time.time() - start_time, 1, 1, (objective(x0_dec_new) - objective(sol["x"])) / objective(sol["x"])])))
                    break
                else:
                    x0_enc = x0_enc_new
                    x0_dec = x0_dec_new

        with open(f'PGD_paillier.csv', 'a') as csvfile:
            np.savetxt(csvfile, results, delimiter=',', fmt='%s', comments='')
