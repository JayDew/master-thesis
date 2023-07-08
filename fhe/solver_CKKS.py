import numpy as np
from Pyfhel import Pyfhel
from scipy.optimize import linprog
from functools import reduce
import time
from util.graphGenerator import GraphGenerator

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
    return np.linalg.pinv(m)


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

# iterations = [479, 86, 145, 76, 256]

for exp in experiments:
    n = exp[0]
    Es = exp[1]
    for E in Es:
        results = np.asarray([np.NAN] * 7)
        for i in range(1,6):  # repeat each experiment 100 times
            # generate random graph
            generator = GraphGenerator(N=n, E=E, seed=i)
            e, c, A = generator.generate_random_graph()
            # c = c / np.linalg.norm(c)  # normalize cost vector
            longest_shortest_path = generator.get_longest_path()
            s = longest_shortest_path[0]  # starting node
            t = longest_shortest_path[-1]  # terminal node
            b = get_b_vector(n, s, t)
            # A = np.asarray([[1, 1], [-1, -1]])
            # b = np.asarray([1, -1])
            # c = np.asarray([1, 2])
            # e = 2
            #################################
            # Exact solution using plaintext
            sol = linprog(c, A_eq=A, b_eq=b)
            opt = sol['fun']
            print('OPT:', opt, '---', sol['x'])
            ###################################

            step_size = 0.0001

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
            # start measuring execution
            start_time = time.time()
            points = []
            convergence = []

            K = 200
            for k in range(K):
                # cloud computes projected gradient descent
                x0_enc_new = _proj(sum_encrypted_vectors(v, gradient(v)))
                # cloud sends back result to client for comparison
                x0_new_pt = decrypt_vector(x0_enc_new)
                # clients locally ensures that values are positive
                x0_new_pt = np.maximum(np.zeros(e), x0_new_pt)
                points.append((x0_new_pt[0], x0_new_pt[1]))

                print(k, objective(x0_new_pt), x0_new_pt)
                x0_new_pt = x0_new_pt + (x0_new_pt - x0_pt) * (k-1)/(k+2)  # we cannot perform this over palintext... (see paper)
                # client encrypts result and sends back to cloud
                v_new = encrypt_vector(x0_new_pt)

                # points.append((x0_new_pt[0], x0_new_pt[1]))
                convergence.append((objective(x0_new_pt) - objective(sol["x"])) / objective(sol["x"])) #remove this after you plot the graphs

                # print(k, 'OPT:', f'{objective(np.rint(x0_new_pt)):.3f}', '---', np.rint(x0_new_pt))
                if np.allclose(x0_pt, x0_new_pt):  # convergence
                    if not (np.isclose(objective(x0_new_pt), objective(sol['x']), rtol=1.e-1) or np.allclose(np.rint(x0_new_pt), sol['x'])):  # correctness
                        results = np.vstack((results, np.asarray([n, e, k, time.time() - start_time, 1, 0,  (objective(x0_new_pt) - objective(sol["x"]))])))
                        fucked_up = True
                        print('we fucked up!')
                        continue
                    else:
                        print(f'convergence after {k - 1} iterations')
                        results = np.vstack((results, np.asarray([n, e, k - 1, time.time() - start_time, 1, 1,  (objective(x0_new_pt) - objective(sol["x"]))])))
                        break
                else:
                    x0_pt = x0_new_pt
                    x0_enc = x0_enc_new
                    v = v_new
            else:
                print('convergence not reached!')
                if np.array_equal(np.rint(x0_new_pt), sol['x']):  # correctness
                    results = np.vstack((results, np.asarray([n, e, k, time.time() - start_time, 0, 1, (objective(x0_new_pt) - objective(sol["x"]))])))
                else:
                    results = np.vstack((results, np.asarray([n, e, k, time.time() - start_time, 0, 0, (objective(x0_new_pt) - objective(sol["x"]))])))

            with open(f'../experiments/final/exp2/CKKS.csv', 'a') as csvfile:
                np.savetxt(csvfile, results, delimiter=',', fmt='%s', comments='')
