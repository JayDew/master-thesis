import matplotlib.pyplot as plt

# data
accelerated_2 = [(8, 1), (30, 25), (48, 41), (72, 33), (90, 63), (138, 57), (142, 62), (242, 124), (248, -1),
                 (354, 141), (376, 191), (438, 80), (536, 382), (624, 248), (724, 736), (772, 135), (884, 154),
                 (918, 468), (1058, 290), (1154, 515), (1280, 248), (1412, 292), (1546, 314), (1684, 479), (1816, 275),
                 (1954, 425), (2024, 298), (2284, 368), (2316, 284), (2420, 442), (2758, 199), (2832, -1), (2922, -1),
                 (3190, 192), (3266, 718), (3486, 662), (3580, 399), (4050, 525)]
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

proximal = [(8, 1), (30, 47), (48, 77), (72, 65), (90, 99), (138, 95), (142, 84), (242, 107), (248, -1), (354, 117),
            (376, 257), (438, 141), (536, 302), (624, 183), (724, 377), (772, 145), (884, 175), (918, 318), (1058, 392),
            (1154, 291), (1280, 378), (1412, 320), (1546, 272), (1684, 492), (1816, 362), (1954, 239), (2024, 382),
            (2284, 304), (2316, 309), (2420, 442), (2758, 199), (2832, -1), (2922, -1), (3190, 192), (3266, 718),
            (3486, 662), (3580, 399), (4050, 525)]

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

edges_prox = [x[0] for x in proximal if x[1] != -1]
iterations_prox = [x[1] for x in proximal if x[1] != -1]
non_conv_prox = [x[0] for x in proximal if x[1] == -1]

# plot the lines
plt.plot(edges_acc_1, iterations_acc_1, linestyle='-', color='C3', label=r'PGD')
plt.plot(edges_acc_1_5, iterations_acc_1_5, linestyle='-', color='C2', label=r'$aPGD \beta$ = 1.5')
plt.plot(edges_acc_1_7, iterations_acc_1_7, linestyle='-', color='C1', label=r'$aPGD \beta$ = 1.7')
plt.plot(edges_acc_2, iterations_acc_2, linestyle='-', color='C0', label=r'$aPGD \beta$ = 2')
plt.plot(edges_prox, iterations_prox, linestyle='-', color='C4', label=r'prox. PGD')

# show the final plot
ax = plt.subplot(111)
ax.set_xlabel('Number of edges')
ax.set_ylabel('Number of iterations')
ax.set_title('Iterations required for convergence \n for calculating the pair-wise longest shorest path')
ax.legend()
ax.grid('on')
plt.xlim([-10, 4300])
plt.ylim([-10, 1700])
plt.show()
