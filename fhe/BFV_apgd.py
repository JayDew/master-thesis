import time
import numpy as np
from Pyfhel import Pyfhel
from functools import reduce
from gmpy2 import gmpy2, mpz
from scipy.optimize import linprog
from util.graphGenerator import GraphGenerator

np.random.seed(420)

HE = Pyfhel()  # Creating empty Pyfhel object
bfv_params = {
    'scheme': 'BFV',    # can also be 'bfv'
    'n': 2**14,         # Polynomial modulus degree, the num. of slots per plaintext,
                        #  of elements to be encoded in a single ciphertext in a
                        #  2 by n/2 rectangular matrix (mind this shape for rotations!)
    't_bits': 60,       # Number of bits in t. Used to generate a suitable value
                        #  for t. Overrides t if specified.
    'sec': 128,         # Security parameter. The equivalent length of AES key in bits.
                        #  Sets the ciphertext modulus q, can be one of {128, 192, 256}
                        #  More means more security but also slower computation.
}
HE.contextGen(**bfv_params)  # Generate context for bfv scheme
HE.keyGen()

DEFAULT_MSGSIZE = 32
DEFAULT_PRECISION = int(DEFAULT_MSGSIZE/2) # of fractional bits
DEFAULT_FACTOR = 10

def fp(scalar, prec=DEFAULT_PRECISION + DEFAULT_FACTOR):
    return int(mpz(gmpy2.mul(float(scalar), 2 ** prec)))

def fp_vector(vec, prec=DEFAULT_PRECISION + DEFAULT_FACTOR):
    return np.asarray([fp(x, prec) for x in vec])

def fp_matrix(mat, prec=DEFAULT_PRECISION + DEFAULT_FACTOR):
    return [fp_vector(x, prec) for x in mat]


def retrieve_fp(scalar, prec=DEFAULT_PRECISION + DEFAULT_FACTOR):
    return scalar / (2 ** prec)


#
def retrieve_fp_vector(vec, prec=DEFAULT_PRECISION + DEFAULT_FACTOR):
    return [retrieve_fp(x, prec) for x in vec]

def retrieve_fp_matrix(mat, prec=DEFAULT_PRECISION + DEFAULT_FACTOR):
    return [retrieve_fp_vector(x, prec) for x in mat]

def encrypt_vector(x):
    return [HE.encrypt(np.asarray([y])) for y in x]

def decrypt_vector(x):
    return [HE.decrypt(i)[0] for i in x]

def encrypt_matrix(x):
    return [[HE.encrypt(y) for y in z] for z in x]

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

def inv(a):
    return np.linalg.pinv(a)

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
            # generate random graph
            generator = GraphGenerator(N=n, E=E, seed=i)
            e, c, A = generator.generate_random_graph()
            # c = c / np.linalg.norm(c)  # normalize cost vector
            longest_shortest_path = generator.get_longest_path()
            s = longest_shortest_path[0]  # starting node
            t = longest_shortest_path[-1]  # terminal node
            b = get_b_vector(n, s, t)
            #################################
            # Exact solution using plaintext
            sol = linprog(c, A_eq=A, b_eq=b)
            opt = sol['fun']
            # print('OPT:', opt, '---', sol['x'])
            ###################################

            step_size = 0.001

            P = np.eye(e) - A.T @ inv(A @ A.T) @ A
            Q = A.T @ inv(A @ A.T) @ b
            x0_pt = np.ones(e) * 0.5

            c_minus_enc = encrypt_vector(fp_vector(step_size * (-c)))
            Q_enc = encrypt_vector(fp_vector(fp_vector(Q)))
            P_enc = encrypt_matrix(fp_matrix(P))
            x0_enc = encrypt_vector(fp_vector(x0_pt))


            def objective(x):
                return c @ x

            def _proj(x):
                return sum_encrypted_vectors(dot_m_encrypted_vectors(x, P_enc), Q_enc)

            def gradient(x):
                return c_minus_enc


            correct_value_after = 0
            correct_found = False

            # start measuring execution
            start_time = time.time()

            k = 0
            while True:
                k = k+1
                print(k)

                # cloud computes projected gradient descent
                x0_enc_new = _proj(sum_encrypted_vectors(x0_enc, gradient(x0_enc)))
                # cloud sends back result to client for comparison
                x0_new_pt = retrieve_fp_vector(retrieve_fp_vector(decrypt_vector(x0_enc_new)))
                # clients locally ensures that values are positive
                x0_new_pt = np.maximum(np.zeros(e), x0_new_pt)
                x0_new_pt = x0_new_pt + (x0_new_pt - x0_pt) * (k-1)/(k+2)  # we cannot perform this over palintext... (see paper)
                # client encrypts result and sends back to cloud
                x0_enc_new = encrypt_vector(fp_vector(x0_new_pt))

                # correctness without convergence
                if np.allclose(np.rint(x0_new_pt), sol['x']) and not correct_found:
                    correct_found = True
                    correct_value_after = k


                if np.allclose(x0_pt, x0_new_pt): #convergence
                    if not (np.isclose(objective(x0_new_pt), objective(sol['x']), rtol=1.e-1) or np.allclose(np.rint(x0_new_pt), sol['x'])):  #correctness
                        results = np.vstack((results, np.asarray([n, e, correct_value_after, k+1,time.time() - start_time, 1, 0, (objective(x0_new_pt) - objective(sol["x"])) / objective(sol["x"])])))
                        fucked_up = True
                        continue
                    print(f'convergence after {k + 1} iterations')
                    results = np.vstack((results, np.asarray([n, e, correct_value_after, k+1, time.time() - start_time, 1, 1, (objective(x0_new_pt) - objective(sol["x"])) / objective(sol["x"])])))
                    print(f'{e}:convergence after {k + 1} iterations')
                    break
                else:
                    x0_pt = x0_new_pt
                    x0_enc = x0_enc_new

                with open(f'BFV.csv', 'a') as csvfile:
                    np.savetxt(csvfile, results, delimiter=',', fmt='%s', comments='')
