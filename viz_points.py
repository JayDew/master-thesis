import matplotlib.pyplot as plt
import numpy as np

OPT_X = [1]
OPT_Y = [0]
plt.scatter(OPT_X, OPT_Y, color='orange', marker='*', s=500)  # plot optimal solution as a star
# Generate a range of x values
x = np.linspace(0, 1, 400)
# Compute the corresponding y values
y = -x + 1
plt.plot(x, y, label='Ax = b')

# data
points_pgd = [(0.5, 0.5), (0.55, 0.45), (0.6, 0.4), (0.65, 0.35), (0.7, 0.3), (0.75, 0.25), (0.8, 0.2), (0.85, 0.15),
              (0.9, 0.1),
              (0.95, 0.05), (1., 0.)]
points_pgd_x = [x[0] for x in points_pgd]
points_pgd_y = [x[1] for x in points_pgd]
temps_pgd = [(0.4, 0.3), (0.45, 0.25), (0.5, 0.2), (0.55, 0.15), (0.6, 0.1), (0.65, 0.05),
             (7.00000000e-01, -1.11022302e-16), (0.75, -0.05), (0.8, -0.1), (0.85, -0.15)]
temps_pgd_x = [x[0] for x in temps_pgd]
temps_pgd_y = [x[1] for x in temps_pgd]

points_a_pgd_2 = [(0.5, 0.5), (0.55, 0.45), (0.65, 0.35), (0.8, 0.2), (1., 0.)]
points_pgd_2_x = [x[0] for x in points_a_pgd_2]
points_pgd_2_y = [x[1] for x in points_a_pgd_2]
temps_2_pgd = [(0.4, 0.3), (0.5, 0.2), (0.65, 0.05), (0.85, -0.15)]
temps_2_pgd_x = [x[0] for x in temps_2_pgd]
temps_2_pgd_y = [x[1] for x in temps_2_pgd]

points_proximal = [(0.5, 0.5), (0.55, 0.45), (0.6, 0.4), (0.6625, 0.3375), (0.7375, 0.2625), (0.825, 0.175),
                   (0.925, 0.075),
                   (1.0075, 0.)]
prximal_temps = [(0.4, 0.3), (0.45, 0.25), (0.5125, 0.1875), (0.5875, 0.1125), (0.675, 0.025), (0.775, -0.075),
                 (0.8875, -0.1875)]

points_prox_x = [x[0] for x in points_proximal]
points_prox_y = [x[1] for x in points_proximal]
temps_prox_pgd_x = [x[0] for x in prximal_temps]
temps_prox_pgd_y = [x[1] for x in prximal_temps]

# plot the points
# plt.scatter(points_pgd_x, points_pgd_y, linestyle='-', color='C3', label=r'PGD')
# plt.scatter(temps_pgd_x, temps_pgd_y, linestyle='-', color='C3')
# plt.scatter(points_pgd_2_x, points_pgd_2_y, linestyle='-', color='C2', label=r'a PGD $\beta$ = 2')
# plt.scatter(temps_2_pgd_x, temps_2_pgd_y, linestyle='-', color='C2')
plt.scatter(points_prox_x, points_prox_y, linestyle='-', color='C1', label=r'prox PGD')
plt.scatter(temps_prox_pgd_x, temps_prox_pgd_y, linestyle='-', color='C1')

# show the final plot
ax = plt.subplot(111)
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_title('Convergence of different PGD algorithms')
ax.legend()
ax.grid('on')
plt.show()
