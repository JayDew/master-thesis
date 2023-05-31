import matplotlib.pyplot as plt

# data
accelerated_2 = [(8, 1), (30, 25), (48, 41), (72, 33), (90, 63), (138, 57), (142, 62), (242, 124), (248, -1),
                 (354, 141), (376, 191), (438, 80), (536, 382), (624, 248), (724, 736), (772, 135), (884, 154),
                 (918, 468), (1058, 290), (1154, 515), (1280, 248), (1412, 292), (1546, 314), (1684, 479), (1816, 275),
                 (1954, 425), (2024, 298), (2284, 368), (2316, 284)]
accelerated_1_7 = [(8, 1), (30, 93), (48, 236), (72, 168), (90, 386), (138, 351), (142, 223), (242, 415), (248, -1),
                   (354, 454), (376, -1), (438, 764), (536, -1), (624, 1210), (724, -1), (772, 760), (884, 971),
                   (918, -1), (1058, -1), (1154, -1), (1280, -1), (1412, -1), (1546, -1), (1684, -1), (1816, -1),
                   (1954, -1), (2024, -1), (2284, -1), (2316, -1)]
accelerated_1_5 = [(8, 1), (30, 152), (48, 390), (72, 276), (90, 641), (138, 582), (142, 370), (242, 689), (248, -1),
                   (354, 754), (376, -1), (438, 1270), (536, -1), (624, -1), (724, -1), (772, 1263), (884, 1613),
                   (918, -1), (1058, -1), (1154, -1), (1280, -1), (1412, -1), (1546, -1), (1684, -1), (1816, -1),
                   (1954, -1), (2024, -1), (2284, -1), (2316, -1)]
accelerated_1 = [(8, 1), (30, 301), (48, 777), (72, 550), (90, 1279), (138, 1162), (142, 739), (242, 1374), (248, -1),
                 (354, 1505), (376, -1), (438, -1), (536, -1), (624, -1), (724, -1), (772, -1), (884, -1), (918, -1),
                 (1058, -1), (1154, -1), (1280, -1), (1412, -1), (1546, -1), (1684, -1), (1816, -1), (1954, -1),
                 (2024, -1), (2284, -1), (2316, -1)]

# parse data
edges_acc_2 = [x[0] for x in accelerated_2 if x[1] != -1]
iterations_acc_2 = [x[1] for x in accelerated_2 if x[1] != -1]
non_conv_acc_2 = [x[0] for x in accelerated_2 if x[1] == -1]

edges_acc_1_7 = [x[0] for x in accelerated_1_7 if x[1] != -1]
iterations_acc_1_7 = [x[1] for x in accelerated_1_7 if x[1] != -1]
non_conv_acc_1_7 = [x[0] for x in accelerated_1_7 if x[1] == -1]

edges_acc_1_5 = [x[0] for x in accelerated_1_5 if x[1] != -1]
iterations_acc_1_5 = [x[1] for x in accelerated_1_5 if x[1] != -1]
non_conv_acc_1_5 = [x[0] for x in accelerated_1_5 if x[1] == -1]

edges_acc_1 = [x[0] for x in accelerated_1 if x[1] != -1]
iterations_acc_1 = [x[1] for x in accelerated_1 if x[1] != -1]
non_conv_acc_1 = [x[0] for x in accelerated_1 if x[1] == -1]

# plot the lines
plt.plot(edges_acc_2, iterations_acc_2, linestyle='-', color='C0', label=r'$\beta$ = 2')
plt.plot(edges_acc_1_7, iterations_acc_1_7, linestyle='-', color='C1', label=r'$\beta$ = 1.7')
plt.plot(edges_acc_1_5, iterations_acc_1_5, linestyle='-', color='C2', label=r'$\beta$ = 1.5')
plt.plot(edges_acc_1, iterations_acc_1, linestyle='-', color='C3', label=r'$\beta$ = 1')

# show the final plot
ax = plt.subplot(111)
ax.set_xlabel('Number of edges')
ax.set_ylabel('Iterations required for convergence')
ax.set_title('')
ax.legend()
ax.grid('on')
plt.show()
