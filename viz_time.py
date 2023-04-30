import matplotlib.pyplot as plt

N = [3, 4, 5, 7, 10, 15, 20, 25, 30]
avg = [0.1, 7.3, 1.1, 1.9, 12.4, 159, 344, 711, 3500]

plt.plot(N, avg, linestyle='-', color='C0')
plt.axhline(y=300, color='r', linestyle=':')

ax = plt.subplot(111)
# set the basic properties
ax.set_xlabel('Number of nodes')
ax.set_ylabel('Time for convergence (seconds)')
ax.set_title('Elapsed time')
ax.grid('on')
plt.show()
