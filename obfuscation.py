from common import get_params, get_data
from dataclasses import dataclass
import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp
from common import f


@dataclass
class Psi:
    """Dataclass encapsulation of instance of the problem"""
    Q: np.ndarray
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray

    def __init__(self, Q: np.ndarray, c: np.ndarray, A: np.ndarray, b: np.ndarray):
        self.Q = Q
        self.c = c
        self.A = A
        self.b = b


class Key:
    """Dataclass encapsulation of private Key"""
    H: np.ndarray
    r: np.ndarray

    def __init__(self, H: np.ndarray, r: np.ndarray):
        self.H = H
        self.r = r


###########

def solve_instance(instance: Psi):
    """This algorithm solves the problem ΨK to produce an output y and a proof Ω."""
    sol = qp(matrix(instance.Q), matrix(instance.c), G=matrix(instance.A), h=matrix(instance.b)
             , options={'show_progress': False})
    y = np.array(sol['x'])
    sigma = None  # TODO! add verification mechanism
    return y


def keyGen(alpha=1000):
    """This algorithm randomly generates a key K with a random security parameter alpha"""
    H = np.random.rand(n * N, m * N) * alpha
    r = np.random.rand(n * N) * alpha
    return Key(H, r)


def probEnc(k: Key, problem: Psi):
    Q = problem.Q
    c = problem.c
    A = problem.A
    b = problem.b
    #
    H = k.H
    r = k.r
    #
    Q_enc = H.T @ Q @ H
    A_enc = A @ H
    b_enc = b + A @ r
    c_enc = c.T @ H - r.T @ Q @ H
    return Psi(Q_enc, c_enc, A_enc, b_enc)


def probDec(k: Key, y: np.ndarray):
    H = k.H
    r = k.r
    return H @ y - r.reshape(-1, 1)


n, m, N = get_params()
Q, c, A, b = get_data()

##### solve the problem in plaintext
problem_plaintext = Psi(Q, c, A, b)
optimal_solution = solve_instance(problem_plaintext)
print("OPTIMAL VALUE ORIGINAL:", f(optimal_solution))
##### solve the problem in encrypted form
key = keyGen()
problem_encrypted = probEnc(key, problem_plaintext)
optimal_solution_enc = solve_instance(problem_encrypted)
optimal_solution_dec = probDec(key, optimal_solution_enc)
print("OPTIMAL VALUE PRIVACY:", f(optimal_solution_dec))
#### check that they are equal
epsilon = 0.05
assert np.abs(np.amin(optimal_solution - optimal_solution_dec)) < epsilon
assert np.abs(np.amax(optimal_solution - optimal_solution_dec)) < epsilon
print('Decryption is correct!')
