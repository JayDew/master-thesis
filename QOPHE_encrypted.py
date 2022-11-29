import numpy as np

from load import get_params, get_data, get_keys, f, save_intermediate_variables
from PaillierHomoVec import *

"""
Securely solving quadratic optimization problem subject to linear constraints.

minimize (1/2)x'Qx + r'x
st       Ex < e
"""

n, m, N = get_params()
Q, c_A, A, b_A = get_data()
pubkey, privkey = get_keys()

b_A_enc = encrypt_vector(pubkey, fp_vector(b_A))
c_A_enc = encrypt_vector(pubkey, fp_vector(c_A))
c_A_enc2 = encrypt_vector(pubkey, fp_vector(np.asarray(fp_vector(c_A))))


def complementary_slackness(mu_opt):
    """
    Given the optimal solution to the dual,
    output the optimal solution to the primal.
    """
    temp = sum_encrypted_vectors(mat_mul(mu_opt, fp_matrix(A.T)), c_A_enc2)
    return mat_mul(temp, fp_matrix((-1) * np.linalg.inv(Q)))


def triangle(mu_enc):
    """
    Compute vector of partial derivatives for
    projected gradient ascent.
    """
    temp = sum_encrypted_vectors(mat_mul(mu_enc, fp_matrix(A.T)), c_A_enc)
    return diff_encrypted_vectors(mat_mul(temp, fp_matrix((-1) * (np.matmul(A, np.linalg.inv(Q))))), b_A_enc)


def mu_bar(mu_enc):
    return sum_encrypted_vectors(mu_enc, mult_vect_by_constant(triangle(mu_enc), fp(nu)))


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


def is_solution_feasible(x):
    """
    Check that the constraints are met.

    Condition E*x < e should hold.
    """
    temp = np.matmul(A, x) < b_A
    print("Number of constraints violations:", np.count_nonzero(temp == False))
    return False if False in temp else True


mu = np.random.sample(n * N)
nu = 0.1

K = 3

for k in range(K):
    mu_enc = encrypt_vector(pubkey, fp_vector(mu))
    mu_bar_ = mu_bar(mu_enc)
    # TODO! include multiplicative binding here
    decrypted = retrieve_fp_vector(
        retrieve_fp_vector(retrieve_fp_vector(retrieve_fp_vector(decrypt_vector(privkey, mu_bar_)))))
    mu_new = np.maximum(decrypted, np.zeros(n * N))
    mu = mu_new

U_opt = np.asarray(retrieve_fp_vector(retrieve_fp_vector(
    retrieve_fp_vector(decrypt_vector(privkey, complementary_slackness(encrypt_vector(pubkey, fp_vector(mu))))))))
print("OPTIMAL VALUE:", f(U_opt))

# Save intermediate values
save_intermediate_variables(U_opt, filename="QOPHE_encrypted")
