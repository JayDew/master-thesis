import time

import numpy as np
from gmpy2 import mpz
from phe import paillier
import gmpy2

DEFAULT_KEYSIZE = 512
DEFAULT_MSGSIZE = 32
DEFAULT_PRECISION = int(DEFAULT_MSGSIZE / 2)  # of fractional bits
DEFAULT_FACTOR = 60

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
    # (15, [210]),
    # (20, [380])
]

for exp in experiments:
    n = exp[0]
    Es = exp[1]
    for E in Es:
        results = np.asarray([np.NAN] * 6)
        for i in range(5):  # repeat each experiment 100 times
            generator = GraphGenerator(N=n, E=E, seed=i)  # generate graph
            e, c, A = generator.generate_random_graph()
            c = c / np.linalg.norm(c)  # normalize cost vector
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

            step_size = 0.1
            enc_c_minus = encrypt_vector(pubkey, fp_vector(step_size * (-c)))
            P = np.eye(e) - A.T @ inv(A @ A.T) @ A
            Q = A.T @ inv(A @ A.T) @ b
            Q_enc = encrypt_vector(pubkey, fp_vector(fp_vector(Q)))
            x0 = np.ones(e) * 0.5
            x0_dec = x0
            x0_enc = encrypt_vector(pubkey, fp_vector(x0))


            def objective(x):
                return c @ x


            def _proj(x):
                return sum_encrypted_vectors(dot_m_encrypted_vectors(x, fp_matrix(P)), Q_enc)


            def gradient(x):
                return enc_c_minus


            # parameters for accelerated
            # projected-gradient method
            y = x0_enc
            # start measuring execution time
            start_time = time.time()
            fucked_up = False

            K = 5000
            for k in range(K):
                if fucked_up:
                    break
                # cloud performs projected gradient descent over encrypted data
                x0_enc_new = _proj(sum_encrypted_vectors(y, gradient(y)))
                # cloud sends data to the client
                # client performs max and combines 2 previous solutions over plaintext
                x0_dec_new = np.maximum(np.zeros(e), retrieve_fp_vector(retrieve_fp_vector(decrypt_vector(privkey, x0_enc_new))))
                x0_dec_new = np.asarray(list(map(lambda x: float(x), x0_dec_new)))
                temp_dec = (x0_dec_new - x0_dec) * (k-1)/(k+2)
                # the client encrypts the result and sends it to the cloud
                x0_enc_new = encrypt_vector(pubkey, fp_vector(x0_dec_new))
                temp_enc = encrypt_vector(pubkey, fp_vector(temp_dec))
                # cloud combines the two previous solutions
                y_new = sum_encrypted_vectors(temp_enc, x0_enc_new)


                # print(k, 'OPT:', f'{objective(np.rint(x0_dec)):.3f}', '---', np.rint(x0_dec))
                if np.allclose(x0_dec, x0_dec_new):  # convergence
                    if not np.array_equal(np.rint(x0_dec_new), sol['x']):  # convergence and correctness
                        results = np.vstack((results, np.asarray([n, e, np.NAN, np.NAN, 1, 0])))
                        fucked_up = True
                        print('we fucked up!')
                        continue
                    print(f'convergence after {k + 1} iterations')
                    results = np.vstack((results, np.asarray([n, e, k + 1, time.time() - start_time, 1, 1])))
                    break
                else:
                    x0_enc = x0_enc_new
                    x0_dec = x0_dec_new
                    y = y_new
            else:
                print('convergence not reached!')
                if np.array_equal(np.rint(x0_dec_new), sol['x']):  # correctness
                    results = np.vstack((results, np.asarray([n, e, np.NAN, np.NAN, 0, 1])))
                else:
                    results = np.vstack((results, np.asarray([n, e, np.NAN, np.NAN, 0, 0])))

        with open(f'FISTA_paillier.csv', 'a') as csvfile:
            np.savetxt(csvfile, results, delimiter=',', fmt='%s', comments='')
