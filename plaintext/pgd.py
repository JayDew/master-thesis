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
    (16, [210]),
    (20, [380])
]

for exp in experiments:
    n = exp[0]
    Es = exp[1]
    for E in Es:
        results = np.asarray([np.NAN] * 8)
        for i in range(50):  # repeat each experiment 50 times
            # generate random graph
            generator = GraphGenerator(N=n, E=E, seed=i)
            e, c, A = generator.generate_random_graph()
            # c_normal = c
            # c = c / np.linalg.norm(c) #normalize cost vector
            longest_shortest_path = generator.get_longest_path() #get the longest shortest path
            s = longest_shortest_path[0]  # starting node
            t = longest_shortest_path[-1]  # terminal node
            b = get_b_vector(n, s, t)
            # generator.save_graph_image(s, t) # save png
            # A = np.asarray([[1, 1], [-1, -1]])
            # b = np.asarray([1, -1])
            # c = np.asarray([1, 2])
            # e = 2
            #################################
            # Exact solution using plaintext
            sol = linprog(c, A_eq=A, b_eq=b)
            opt = sol['fun']
            # print('OPT:', opt, '---', sol['x'])
            ###################################

            step_size = 0.001
            P = np.eye(e) - A.T @ inv(A @ A.T) @ A
            Q = A.T @ inv(A @ A.T) @ b
            x0 = np.ones(e) * 0.5  # initial guess

            def objective(x):
                return c @ x


            def gradient(x):
                return c

            # start measuring execution
            start_time = time.time()

            convergence = []
            fucked_up = False
            correct_value_after = 0
            correct_found = False

            k = 0
            while True:
                k = k + 1

                # if (k % 10 == 0):
                #     step_size = step_size / 2

                if fucked_up:
                    break
                x0_new = P @ (x0 - step_size * gradient(x0)) + Q
                x0_new = np.maximum(np.zeros(e), x0_new)

                convergence.append((objective(x0_new) - objective(sol["x"])) / objective(sol["x"])) #remove this after you plot the graphs


                # correctness without convergence
                if np.allclose(np.rint(x0_new), sol['x']) and not correct_found:
                    correct_found = True
                    correct_value_after = k

                if np.allclose(x0, x0_new):  # convergence
                    if not (np.isclose(objective(x0_new), objective(sol['x']), rtol=1.e-2) or np.allclose(np.rint(x0_new), sol['x'])):  # correctness
                        results = np.vstack((results, np.asarray([n, e, correct_value_after, k + 1, time.time() - start_time, 1, 0, (objective(x0_new) - objective(sol["x"]))])))
                        fucked_up = True
                        continue
                    print(f'convergence after {k + 1} iterations')
                    results = np.vstack((results, np.asarray([n, e, correct_value_after, k + 1, time.time() - start_time, 1, 1, (objective(x0_new) - objective(sol["x"])) / objective(sol["x"])])))
                    break
                else:
                    x0 = x0_new

        with open(f'pgd.csv', 'a') as csvfile:
            np.savetxt(csvfile, results, delimiter=',', fmt='%s', comments='')
