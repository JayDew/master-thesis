import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp

from common import get_params, get_data, save_intermediate_variables

n, m = get_params()
Q, c, A, b = get_data()

# Solve to optimality using cvxopt
sol = qp(matrix(Q), matrix(c), matrix(A), matrix(b), options={'show_progress': False})
U_opt = np.array(sol['x'])
print("OPTIMAL VALUE:", sol['primal objective'])

# Save intermediate values
save_intermediate_variables(U_opt, filename="cvxopt")
