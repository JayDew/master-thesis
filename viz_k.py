import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

df = pd.read_csv('res/results.csv')
# Fill NaN values with column averages
df = df.fillna(df.mean())

# get the M values
M = df["M"].unique()
df = df.set_index('M')

K_avg = np.zeros(len(M))
K_SD = np.zeros(len(M))

# get the average and standard deviation
for idx, m in enumerate(M):
    K = np.array(df.loc[m, 'k'])
    K_avg[idx] = np.mean(K)
    K_SD[idx] = np.std(K)

# smooth the curve
# 300 represents number of points to make between M.min and M.max
xnew = np.linspace(M.min(), M.max(), 300)
spl = make_interp_spline(M, K_avg, k=3)
spl_err = make_interp_spline(M, K_SD, k=3)
power_smooth = spl(xnew)
power_smooth_err = spl_err(xnew)

# last few points with use a different lambda
plt.plot(xnew[:-15], power_smooth[:-15], linestyle='-', color='C0', label=r'$\alpha = 0.01$')
plt.plot(xnew[-15:], power_smooth[-15:], linestyle='--', color='C0', label=r'$\alpha = 0.005$')
plt.fill_between(xnew, power_smooth - power_smooth_err / 2, power_smooth + power_smooth_err / 2, alpha=0.2,
                 facecolor='#089FFF', linewidth=4, linestyle='dashdot',
                 antialiased=True)

plt.legend(loc="upper left", fontsize=15)
ax = plt.subplot(111)
ax.set_xlabel('Number of edges')
ax.set_ylabel('Number of iterations for convergence')
ax.set_title('Average Number of Iterations for Convergence')
ax.grid('on')
plt.show()
