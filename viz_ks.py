import numpy as np
import matplotlib.pyplot as plt

x = np.arange(50)
y = [(k - 2) / (k + 1) for k in x]
plt.scatter(x, y, color="red", marker="x", alpha=0.5)

ax = plt.subplot(111)
ax.set_xlabel('k')
ax.set_ylabel('(k-2)/(k+1)')
plt.show()
