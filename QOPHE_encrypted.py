from common import *
from PaillierHomoVec import *

"""
Securely solving quadratic optimization problem subject to linear constraints.

minimize (1/2)x'Qx + c_A'x
st       Ax < b_A
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
    return diff_encrypted_vectors(mat_mul(temp, fp_matrix((-1) * (A @ np.linalg.inv(Q)))), b_A_enc)


def mu_bar(mu_enc):
    """
    Projected gradient ascent using encrypted dual .
    """
    return sum_encrypted_vectors(mu_enc, mult_vect_by_constant(triangle(mu_enc), fp(nu)))


def triangle_squared():
    """
    Hessian of the Lagrange-dual function.
    """
    return (-1) * np.matmul(np.matmul(A, np.linalg.inv(Q)), A.T)


mu = np.random.rand(n * N)
nu = 0.6
K = 1000

# iteratively update the dual variables
for k in range(K):
    print(k)
    mu_enc = encrypt_vector(pubkey, fp_vector(mu))
    # Server securely computes gradient ascent
    mu_bar_enc = mu_bar(mu_enc)
    # Server sends mu_bar_ to the Client
    # Client decrypts mu_bar_
    mu_bar_dec = retrieve_fp_vector(
        retrieve_fp_vector(retrieve_fp_vector(retrieve_fp_vector(decrypt_vector(privkey, mu_bar_enc)))))
    # Client projects the dual variables
    mu_new = np.maximum(mu_bar_dec, np.zeros(n * N))
    # And sends it back to the server
    mu = mu_new

    U_opt = np.asarray(retrieve_fp_vector(retrieve_fp_vector(
        retrieve_fp_vector(decrypt_vector(privkey, complementary_slackness(encrypt_vector(pubkey, fp_vector(mu))))))))
    print("OPTIMAL VALUE:", f(U_opt))

U_opt = np.asarray(retrieve_fp_vector(retrieve_fp_vector(
    retrieve_fp_vector(decrypt_vector(privkey, complementary_slackness(encrypt_vector(pubkey, fp_vector(mu))))))))
print("OPTIMAL VALUE:", f(U_opt))
# assert is_solution_feasible(U_opt), "Make sure that our solution is feasible!"

# Save intermediate values
save_intermediate_variables(U_opt, filename="QOPHE_encrypted")
