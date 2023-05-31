import numpy as np
from scipy.optimize import linprog
from graphGenerator import GraphGenerator

np.random.seed(420)


def inv(A, lamb=0.1):
    # Compute the Tikhonov pseudo-inverse of A
    return np.linalg.inv(A.T.dot(A) + lamb * np.eye(A.shape[1])).dot(A.T)
    # Compute the Moore-Penrose pseudo-inverse of A
    # return np.linalg.pinv(A)


def get_b_vector(N, s, t):
    b = np.zeros(N)
    b[s] = 1
    b[t] = -1
    return b


# generate random graph
Ns = [5, 10, 15, 20, 25, 30, 35, 40]
# number of nodes
p = 0.1  # probability of two edges being connected

K_longest_shortest_path = []

for N in Ns:
    generator = GraphGenerator(N=N, p=p)  # generate graph
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

    step_size = 0.0005  # or 0.0001
    P = np.eye(e) - A.T @ inv(A @ A.T) @ A
    Q = A.T @ inv(A @ A.T) @ b
    x0 = np.ones(e) * 0.5


    def objective(x):
        return c @ x


    def gradient(x):
        return c


    K = 2000
    for k in range(K):
        x0 = P @ (x0 - step_size * gradient(x0)) + Q
        x0 = np.maximum(np.zeros(e), x0)
        # print(k, 'OPT:', f'{objective(np.rint(x0)):.3f}', '---', np.rint(x0))

        if np.array_equal(np.rint(x0), sol['x']):
            print(f'convergence after {k + 1} iterations')
            K_longest_shortest_path.append((e, k + 1))
            break
    else:
        print('convergence not reached!')
        K_longest_shortest_path.append((e, -1))

print(K_longest_shortest_path)
print()
