from common import *

"""
Solving quadratic optimization problem subject to 
linear constraints.
"""

n, m, N = get_params()
H, r, E, e = get_data()

e = e.reshape(-1, 1)
r = r.reshape(-1, 1)


def g(mu):
    """
    Dual function that we want to maximize
    """
    temp = np.matmul(E.T, mu) + r
    result = -0.5 * np.matmul(temp.T,
                              np.linalg.inv(H).dot(temp)) - np.matmul(mu.T,
                                                                      e)
    return result


def complementary_slackness(mu_opt):
    """
    Given the optimal solution to the dual,
    output the optimal solution to the primal.
    """
    return (-1) * np.matmul(np.linalg.inv(H), (E.T.dot(mu_opt) + r))


def triangle(mu):
    """
    Compute vector of partial derivatives for
    projected gradient ascent.
    """
    temp = E.T.dot(mu) + r
    return (-1) * (np.matmul(E, np.linalg.inv(H))).dot(temp) - e


def triangle_squared():
    """
    Hessian of the Lagrange-dual function.
    """
    return (-1) * np.matmul(np.matmul(E, np.linalg.inv(H)), E.T)


def my_max(z):
    """
    Ensures that elements of mu cannot be negative.
    """
    return max(0, z)


mu_new = np.random.sample(n * N).reshape(-1, 1)
nu = triangle_squared().max()
epsilon = 10 ** (-6)
condition = True

while condition:
    mu_old = mu_new
    mu_new = np.maximum(mu_old + nu * triangle(mu_old), np.zeros(n * N).reshape(-1, 1))
    if np.linalg.norm(mu_new - mu_old) <= epsilon:
        condition = False
else:
    mu_new = np.vectorize(my_max)(mu_new + nu * triangle(mu_new))

U_opt = complementary_slackness(mu_opt=mu_new)
print("OPTIMAL VALUE:", f(U_opt))
assert is_solution_feasible(U_opt), "Make sure that our solution is feasible!"

# Save intermediate values
save_intermediate_variables(U_opt, filename="QOPHE")
