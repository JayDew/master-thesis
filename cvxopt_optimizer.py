import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp

from common import get_params, get_data, save_intermediate_variables

n, m, N = get_params()
H, r, E, e = get_data()

# Solve to optimality using cvxopt
sol = qp(matrix(H), matrix(r), matrix(E), matrix(e), options={'show_progress': False})
U_opt = np.array(sol['x'])
print("OPTIMAL VALUE:", sol['primal objective'])

# Save intermediate values
save_intermediate_variables(U_opt, filename="cvxopt")
