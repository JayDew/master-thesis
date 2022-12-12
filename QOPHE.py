from common import *

"""
Solving quadratic optimization problem subject to 
linear constraints.
"""

n, m = get_params()
Q, c, A, b = get_data()

b = b.reshape(-1, 1)
c = c.reshape(-1, 1)


def g(mu):
    """
    Dual function that we want to maximize
    """
    temp = np.matmul(A.T, mu) + c
    result = -0.5 * np.matmul(temp.T,
                              np.linalg.inv(Q).dot(temp)) - np.matmul(mu.T,
                                                                      b)
    return result


def complementary_slackness(mu_opt):
    """
    Given the optimal solution to the dual,
    output the optimal solution to the primal.
    """
    return (-1) * np.matmul(np.linalg.inv(Q), (A.T.dot(mu_opt) + c))


def triangle(mu):
    """
    Compute vector of partial derivatives for
    projected gradient ascent.
    """
    temp = A.T.dot(mu) + c
    return (-1) * (np.matmul(A, np.linalg.inv(Q))).dot(temp) - b


def triangle_squared():
    """
    Hessian of the Lagrange-dual function.
    """
    return (-1) * np.matmul(np.matmul(A, np.linalg.inv(Q)), A.T)


def my_max(z):
    """
    Ensures that elements of mu cannot be negative.
    """
    return max(0, z)


mu_new = np.zeros(n).reshape(-1, 1)
nu = 0.3
epsilon = 10 ** (-3)
condition = True

counter = 0
while condition:
    counter += 1
    mu_old = mu_new
    mu_new = np.maximum(mu_old + nu * triangle(mu_old), np.zeros(n).reshape(-1, 1))
    if np.linalg.norm(mu_new - mu_old) <= epsilon:
        condition = False
    # print(counter)
    # U_opt = complementary_slackness(mu_opt=mu_new)
    # print("OPTIMAL VALUE:", f(U_opt))
else:
    mu_new = np.vectorize(my_max)(mu_new + nu * triangle(mu_new))

U_opt = complementary_slackness(mu_opt=mu_new)
print("OPTIMAL VALUE:", f(U_opt))
# assert is_solution_feasible(U_opt), "Make sure that our solution is feasible!"

# Save intermediate values
save_intermediate_variables(U_opt, filename="QOPHE")
