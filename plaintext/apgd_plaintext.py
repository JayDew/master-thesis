import numpy as np
from scipy.optimize import linprog
from util.graphGenerator import GraphGenerator
import time

np.random.seed(420)

def inv(A):
    # Moore-Penrose pseudo-inverse of A
    return np.linalg.pinv(A)


def get_b_vector(N, s, t):
    b = np.zeros(N)
    b[s] = 1
    b[t] = -1
    return b


experiments = [
    (5, [20]),
    (8, [56]),
    (10, [90]),
    (15, [210]),
    (20, [380])
]

for exp in experiments:
    n = exp[0]
    Es = exp[1]
    for E in Es:
        results = np.asarray([np.NAN] * 6)
        for i in range(100):  # repeat each experiment 100 times
            # generate random graph
            generator = GraphGenerator(N=n, E=E, seed=i)
            e, c, A = generator.generate_random_graph()
            c = c / np.linalg.norm(c) #normalize cost vector
            longest_shortest_path = generator.get_longest_path() #get the longest shortest path
            s = longest_shortest_path[0]  # starting node
            t = longest_shortest_path[-1]  # terminal node
            b = get_b_vector(n, s, t)
            # generator.save_graph_image(s, t) # save png
            #################################
            # Exact solution using plaintext
            sol = linprog(c, A_eq=A, b_eq=b)
            opt = sol['fun']
            # print('OPT:', opt, '---', sol['x'])
            ###################################

            step_size = 0.1
            P = np.eye(e) - A.T @ inv(A @ A.T) @ A
            Q = A.T @ inv(A @ A.T) @ b
            x0 = np.ones(e) * 0.5  # initial guess

            def objective(x):
                return c @ x


            def gradient(x):
                return c


            # parameters for accelerated
            # projected-gradient method
            v = x0
            beta = 2  # set beta = 1 for normal PGD
            # start measuring execution
            start_time = time.time()

            temps = []
            points = []
            convergence = []
            fucked_up = False

            K = 5000
            for k in range(K):
                if fucked_up:
                    break
                temp = v - step_size * gradient(v)
                x0_new = P @ (temp) + Q
                x0_new = np.maximum(np.zeros(e), x0_new)
                v_new = x0 + beta * (x0_new - x0)
                # convergence.append((objective(x0_new) - objective(sol["x"])) / objective(sol["x"]))
                # print(k, 'OPT:', f'{objective(np.rint(x0)):.3f}', '---', np.rint(x0))

                if np.allclose(x0, x0_new):  # convergence
                    if not np.array_equal(np.rint(x0_new), sol['x']):  # correctness
                        results = np.vstack((results, np.asarray([n, e, np.NAN, np.NAN, 1, 0])))
                        fucked_up = True
                        print('we fucked up!')
                        continue
                    print(f'convergence after {k + 1} iterations')
                    results = np.vstack((results, np.asarray([n, e, k + 1, time.time() - start_time, 1, 1])))
                    break
                else:
                    x0 = x0_new
                    v = v_new
            else:
                print('convergence not reached!')
                if np.array_equal(np.rint(x0_new), sol['x']):  # correctness
                    results = np.vstack((results, np.asarray([n, e, np.NAN, np.NAN, 0, 1])))
                else:
                    results = np.vstack((results, np.asarray([n, e, np.NAN, np.NAN, 0, 0])))

        with open(f'apgd_beta_{beta}.csv', 'a') as csvfile:
            np.savetxt(csvfile, results, delimiter=',', fmt='%s', comments='')
