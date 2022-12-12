import numpy as np
import matplotlib.pyplot as plt
from common import get_params

n, m = get_params()

######## creating the TIME plot
cvxopt = np.load(f'Temp/cvxopt_{n}_{m}_time.npy')
alexandru = np.load(f'Temp/alexandru_{n}_{m}_time.npy')
wang = np.load(f'Temp/wang_{n}_{m}_time.npy')

data = {'optimum': cvxopt, 'Alexandru et al.': alexandru, 'Wang et al.': wang}
# creating the bar plot
plt.bar(list(data.keys()), list(data.values()), color='maroon', width=0.4)
# convert y-axis to Logarithmic scale
plt.yscale("log")
plt.ylabel("Time (seconds) - log scale")
plt.title(f'Computation time on {n}x{m} matrix')
plt.show()
######## end TIME plot


######## creating PERFORMANCE plot
cvxopt = np.load(f'Temp/cvxopt_{n}_{m}.npy')[0][0]
alexandru = np.load(f'Temp/alexandru_{n}_{m}.npy')
wang = np.load(f'Temp/wang_{n}_{m}.npy')[0][0]

data = {'optimum': abs(cvxopt), 'Alexandru et al.': abs(alexandru), 'Wang et al.': abs(wang)}
# creating the bar plot
plt.bar(list(data.keys()), list(data.values()), color='blue', width=0.4)
# convert y-axis to Logarithmic scale
# plt.yscale("log")
plt.ylabel("Optimum value")
plt.title(f'Optimal value on {n}x{m} matrix')
plt.show()
######## end PERFORMANCE plot
