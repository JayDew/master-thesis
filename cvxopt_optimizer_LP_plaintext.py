import numpy as np
from scipy.optimize import linprog
from graphGenerator import GraphGenerator

np.random.seed(420)


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
N = 30  # number of nodes
p = 0.1  # probability of two edges being connected

s = 0  # hardcoded starting node
t = 13  # hardcoded terminal node between 1 and N-1

generator = GraphGenerator(N=N, p=p)  # generate graph
e, c, A = generator.generate_random_graph()

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


def objective(x):
    return c @ x


def gradient(x):
    return c


K = 500
for k in range(K):
    x0 = P @ (x0 - step_size * c) + Q
    x0 = np.maximum(np.zeros(e), x0)
    print(k, 'OPT:', f'{objective(np.rint(x0)):.3f}', '---', np.rint(x0))

    if np.array_equal(np.rint(x0), sol['x']):
        print(f'convergence after {k + 1} iterations')
        break
else:
    print('convergence not reached!')
