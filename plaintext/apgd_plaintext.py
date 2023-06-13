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
        results = np.asarray([np.NAN] * 7)
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
            v_new = x0
            beta = 2  # set beta = 1 for normal PGD
            # start measuring execution
            start_time = time.time()

            temps = []
            points = np.zeros(e)
            convergence = []
            fucked_up = False

            K = 3000
            for k in range(K):
                if fucked_up:
                    break
                x0_new = P @ (v_new - step_size * gradient(v_new)) + Q
                x0_new = np.maximum(np.zeros(e), x0_new)
                v = x0 + beta * (x0_new - x0)

                if np.allclose(x0, x0_new):  # convergence
                    if not np.equal(objective(np.rint(x0_new)), objective(sol['x'])):  # correctness
                        results = np.vstack((results, np.asarray([n, e, np.NAN, np.NAN, 1, 0, ((objective(x0_new) - objective(sol["x"])) / objective(sol["x"]))])))
                        fucked_up = True
                        print('we fucked up!')
                        continue
                    print(f'convergence after {k + 1} iterations')
                    results = np.vstack((results, np.asarray([n, e, k + 1, time.time() - start_time, 1, 1, ((objective(x0_new) - objective(sol["x"])) / objective(sol["x"]))])))
                    break
                else:
                    x0 = x0_new
                    v_new = v
            else:
                print('convergence not reached!')
                if np.equal(objective(np.rint(x0_new)), objective(sol['x'])):  # correctness
                    results = np.vstack((results, np.asarray([n, e, np.NAN, np.NAN, 0, 1, ((objective(x0_new) - objective(sol["x"])) / objective(sol["x"]))])))
                else:
                    results = np.vstack((results, np.asarray([n, e, np.NAN, np.NAN, 0, 0, ((objective(x0_new) - objective(sol["x"])) / objective(sol["x"]))])))

        with open(f'apgd_beta_{beta}.csv', 'a') as csvfile:
            np.savetxt(csvfile, results, delimiter=',', fmt='%s', comments='')
