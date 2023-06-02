import numpy as np

points_pgd = np.asarray(
    [[0.55, 0.45], [0.6, 0.4], [0.65, 0.35], [0.7, 0.3], [0.75, 0.25], [0.8, 0.2], [0.85, 0.15], [0.9, 0.1],
     [0.95, 0.05], [1., 0.]])

points_a_pgd_2 = np.asarray([[0.5, 0.5], [0.55, 0.45], [0.65, 0.35], [0.8, 0.2], [1., 0.]])

points_proximal = np.asarray(
    [[0.55, 0.45], [0.6, 0.4], [0.6625, 0.3375], [0.7375, 0.2625], [0.825, 0.175], [0.925, 0.075],
     [1.0075, 0.]])

# # check distance
points = points_pgd

for i in range(len(points) - 1):
    current = points[i]
    next = points[i + 1]
    print(np.linalg.norm(current - next))
