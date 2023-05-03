import numpy as np
from Pyfhel import Pyfhel
from scipy.optimize import linprog
from functools import reduce
import time
import random
from graphGenerator import GraphGenerator

np.random.seed(420)

HE = Pyfhel()  # Creating empty Pyfhel object
ckks_params = {
    'scheme': 'CKKS',
    'n': 2 ** 14,  # Polynomial modulus degree
    'scale': 2 ** 30,  # All the encodings will use it for float->fixed point
    'qi_sizes': [60, 30, 30, 30, 60]  # Number of bits of each prime in the chain
}
HE.contextGen(**ckks_params)  # Generate context for ckks scheme
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
    # return np.linalg.pinv(m)


def get_b_vector(N, s, t):
    b = np.zeros(N)
    b[s] = 1
    b[t] = -1
    return b


# generate random graph
Ns = [3]  # number of nodes
p = 0.1  # probability of two edges being connected

for N in Ns:
    s = 0  # hardcoded starting node
    generator = GraphGenerator(N=N, p=p)  # generate graph
    e, c, A = generator.generate_random_graph()

    for _ in range(5):  # repeat all experiments 5 times
        t = random.randint(1, N - 1)  # random terminal node
        b = get_b_vector(N, s, t)
        #################################
        # Exact solution using plaintext
        sol = linprog(c, A_eq=A, b_eq=b)
        opt = sol['fun']
        print('OPT:', opt, '---', sol['x'])

        ###################################

        step_size = 0.005

        P = np.eye(e) - A.T @ inv(A @ A.T) @ A
        Q = A.T @ inv(A @ A.T) @ b

        x0 = np.ones(e) * 0.5
        x0_enc = encrypt_vector(x0)
        c_enc = encrypt_vector(c)
        c_minus_enc = encrypt_vector((step_size * (-c)))
        b_enc = encrypt_vector(b)
        P_enc = encrypt_matrix(P)
        Q_enc = encrypt_vector(Q)


        def _proj(x):
            return sum_encrypted_vectors(dot_m_encrypted_vectors(x, P_enc), Q_enc)


        def objective(x):
            return c @ x


        def gradient(x):
            return c_minus_enc


        start = time.time()

        K = 500
        for k in range(K):
            # server computes projected gradient descent
            u_enc = _proj(sum_encrypted_vectors(x0_enc, gradient(x0_enc)))
            x0 = P @ (x0 - c * step_size) + Q
            # server sends back result to client for comparison
            u_dec = decrypt_vector(u_enc)
            # clients locally ensures that values are positive
            x0_new = np.maximum(np.zeros(e), u_dec)
            x0 = np.maximum(np.zeros(e), x0)
            # client encrypts result and sends back to server
            mu_enc = encrypt_vector(x0_new)
            # the client receives the final result
            mu_dec = decrypt_vector(mu_enc)
            opt_privacy = objective(np.rint(mu_dec))
            print(k, 'OPT:', f'{opt_privacy:.3f}', '---', np.rint(mu_dec))

            # experiment 1 - check number of iterations until convergence
            if np.array_equal(np.rint(mu_dec), sol['x']):
                print('optimal solution after ', {k}, 'iterations')
                end = time.time()
                generator.save_to_csv(N, e, s, t, k, end - start, 1)
                break

        else:
            end = time.time()
            generator.save_to_csv(N, e, s, t, K, end - start, 0)

        # save for processing
        dic = {}
        dic['pred'] = np.rint(decrypt_vector(x0_enc))
        dic['true'] = sol['x']
        FOLDER = 'res'
        np.savez(f'{FOLDER}/{N}_{s}_{t}.npz', **dic)
