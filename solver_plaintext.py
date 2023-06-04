import numpy as np
from scipy.optimize import linprog
from graphGenerator import GraphGenerator
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


# number of nodes - from 5 to 50 with increments of 1
# Ns = np.arange(5, 50, 1, dtype=int)
Ns = [20]

K_longest_shortest_path = []

for N in Ns:
    # generate random graph
    generator = GraphGenerator(N=N)
    e, c, A = generator.generate_random_graph()
    longest_shortest_path = generator.get_longest_path()
    s = longest_shortest_path[0]  # starting node
    t = longest_shortest_path[-1]  # terminal node
    b = get_b_vector(N, s, t)
    # generator.save_graph_image(s, t)
    #################################
    # Exact solution using plaintext
    sol = linprog(c, A_eq=A, b_eq=b)
    opt = sol['fun']
    # print('OPT:', opt, '---', sol['x'])

    ###################################

    step_size = 0.00007  # or 0.0001
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

    K = 20000
    for k in range(K):
        temp = v - step_size * gradient(v)
        x0_new = P @ (temp) + Q
        x0_new = np.maximum(np.zeros(e), x0_new)
        v_new = x0 + beta * (x0_new - x0)
        # convergence.append((objective(x0_new) - objective(sol["x"])) / objective(sol["x"]))
        # print(k, 'OPT:', f'{objective(np.rint(x0)):.3f}', '---', np.rint(x0))

        if np.allclose(x0, x0_new) and np.array_equal(np.rint(x0_new), sol['x']): # convergence and correctness
            print(f'{e}:convergence after {k + 1} iterations')
            K_longest_shortest_path.append((e, k + 1, time.time() - start_time))
            break
        else:
            x0 = x0_new
            v = v_new
            temps.append((temp[0], temp[1]))
            points.append((x0[0], x0[1]))
    else:
        print('convergence not reached!')
        K_longest_shortest_path.append((e, -1))

print(K_longest_shortest_path)
print(convergence)
