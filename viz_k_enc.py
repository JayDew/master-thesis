import matplotlib.pyplot as plt

# data
plaintext = sorted_array = sorted([(8, 1), (12, 18), (14, 11), (16, 13), (22, 10), (26, 35), (30, 23), (32, 21), (34, 26), (30, 13), (52, 42), (48, 30), (64, 28), (60, 22), (66, 44), (64, 115), (70, 59), (88, 90), (76, 38), (92, 53), (106, 45), (106, 42), (110, 59), (122, 164), (146, 53)], key=lambda x: x[0])
pailler= sorted([(8, 1), (12, 9), (14, 8), (16, 8), (22, 9), (26, 11), (30, 22), (32, 20), (34, 11), (30, 11), (52, 30), (48, 30), (64, 26), (60, 18), (66, 41), (64, 56), (70, 59), (88, 88), (76, 33)], key=lambda x: x[0])

LIMIT = 150

# parse data
edges_plaintext = [x[0] for x in plaintext if x[1] != -1 and x[0] < LIMIT]
iterations_plaintext = [x[1] for x in plaintext if x[1] != -1 and x[0] < LIMIT]

edges_pailler = [x[0] for x in pailler if x[1] != -1 and x[0] < LIMIT]
iterations_pailler = [x[1] for x in pailler if x[1] != -1 and x[0] < LIMIT]


# plot the lines
plt.plot(edges_plaintext, iterations_plaintext, linestyle='-', color='C0', label=r'plaintext')
plt.plot(edges_pailler, iterations_pailler, linestyle='-', color='C1', label=r'PHE')

# show the final plot
ax = plt.subplot(111)
ax.set_xlabel('Number of edges')
ax.set_ylabel('Number of iterations')
ax.set_title('Iterations required for convergence \n for calculating the pair-wise longest shorest path')
ax.legend()
ax.grid('on')
plt.show()
