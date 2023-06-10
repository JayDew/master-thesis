import numpy as np
from Pyfhel import Pyfhel
from scipy.optimize import linprog
from functools import reduce
import time
from util.util import GraphGenerator

np.random.seed(420)

HE = Pyfhel()  # Creating empty Pyfhel object
ckks_params = {
    'scheme': 'CKKS',
    'n': 2 ** 15,  # Polynomial modulus degree
    'scale': 2 ** 30,  # All the encodings will use it for float->fixed point
    'qi_sizes': [60, 30, 30, 30, 60]  # Number of bits of each prime in the chain
}
HE.contextGen(**ckks_params)  # Generate context for ckks scheme
HE.keyGen()  # Key Generation: generates a pair of public/secret keys


def encrypt_vector(x):
    return [HE.encrypt(y) for y in x]


def encrypt_matrix(x):
    return [[HE.encrypt(y) for y in z] for z in x]


def decrypt_vector(x):
    return [HE.decrypt(i)[0] for i in x]


def sum_encrypted_vectors(x, y):
    return [x[i] + y[i] for i in range(np.size(x))]

def diff_encrypted_vectors(x, y):
    return [x[i] - y[i] for i in range(np.size(x))]

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


# number of nodes - from 5 to 50 with increments of 1
Ns = np.arange(5, 20, 1, dtype=int)

K_longest_shortest_path = []

for N in Ns:
    # generate random graph
    generator = GraphGenerator(N=N)
    e, c, A = generator.generate_random_graph()
    longest_shortest_path = generator.get_longest_path()
    s = longest_shortest_path[0]  # starting node
    t = longest_shortest_path[-1]  # terminal node
    b = get_b_vector(N, s, t)
    #################################
    # Exact solution using plaintext
    sol = linprog(c, A_eq=A, b_eq=b)
    opt = sol['fun']
    print('OPT:', opt, '---', sol['x'])

    ###################################

    step_size = 0.00007

    P = np.eye(e) - A.T @ inv(A @ A.T) @ A
    Q = A.T @ inv(A @ A.T) @ b

    x0_pt = np.ones(e) * 0.5
    x0_enc = encrypt_vector(x0_pt)
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


    v = x0_enc
    beta = 2  # set beta = 1 for normal PGD
    # start measuring execution
    start_time = time.time()

    K = 1000

    for k in range(K):
        # cloud computes projected gradient descent
        temp = sum_encrypted_vectors(v, gradient(v))
        x0_enc_new = _proj(temp)
        # cloud sends back result to client for comparison
        x0_new_pt = decrypt_vector(x0_enc_new)
        # clients locally ensures that values are positive
        x0_new_pt = np.maximum(np.zeros(e), x0_new_pt)
        # client encrypts result and sends back to cloud
        x0_enc_new = encrypt_vector(x0_new_pt)
        # the cloud receives the final result
        v_new = sum_encrypted_vectors(x0_enc, mul_sc_encrypted_vectors(diff_encrypted_vectors(x0_enc_new, x0_enc), np.ones(e) * beta))

        print(k, 'OPT:', f'{objective(np.rint(x0_new_pt)):.3f}', '---', np.rint(x0_new_pt))

        if np.array_equal(np.rint(x0_new_pt), sol['x']):  # convergence and correctness
            print('optimal solution after ', {k+1}, 'iterations')
            end = time.time()
            K_longest_shortest_path.append((e, k + 1, time.time() - start_time))
            break
        else:
            x0_pt = x0_new_pt
            x0_enc = x0_enc_new
            v = v_new
    else:
        print('convergence not reached!')
        K_longest_shortest_path.append((e, -1))