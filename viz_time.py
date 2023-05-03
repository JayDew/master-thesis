import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline
import numpy as np

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

T_avg = np.zeros(len(M))
T_SD = np.zeros(len(M))

# get the average and standard deviation
for idx, m in enumerate(M):
    K = np.array(df.loc[m, 'time'])
    T_avg[idx] = np.mean(K)
    T_SD[idx] = np.std(K)

# smooth the curve
# 300 represents number of points to make between T.min and T.max
xnew = np.linspace(M.min(), M.max(), 300)
spl = make_interp_spline(M, T_avg, k=3)
spl_err = make_interp_spline(M, T_SD, k=3)
power_smooth = spl(xnew)
power_smooth_err = spl_err(xnew)

plt.plot(xnew, power_smooth, linestyle='-', color='C0')
plt.axhline(y=300, color='r', linestyle=':')

plt.fill_between(xnew, power_smooth - power_smooth_err / 2, power_smooth + power_smooth_err / 2, alpha=0.2,
                 facecolor='#089FFF', linewidth=4, linestyle='dashdot',
                 antialiased=True)

ax = plt.subplot(111)
ax.set_xlabel('Number of edges')
ax.set_ylabel('Time for convergence (seconds)')
ax.set_title('Average Elapsed time')
ax.grid('on')
plt.show()
