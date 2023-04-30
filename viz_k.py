import matplotlib.pyplot as plt

# plt.style.use('seaborn-whitegrid')


N = [3, 4, 5, 7, 10, 15, 20, 25, 30]
K_avg = [1, 1, 3.2, 2.6, 9.2, 26.6, 41, 51.6, 100]

# plt.plot(N[:-1], K_avg[:-1], linestyle='-', color='C0')
plt.plot(N, K_avg, linestyle='-', color='C0')

ax = plt.subplot(111)

# set the basic properties
ax.set_xlabel('Number of nodes')
ax.set_ylabel('Number of iterations for convergence')
ax.set_title('Average Number of Iterations for Convergence')
ax.grid('on')
plt.show()