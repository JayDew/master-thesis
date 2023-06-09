import pandas as pd
import matplotlib.pyplot as plt

beta = 2
df = pd.read_csv(f'foo_beta_{beta}.csv')
df.columns = ['n', 'e', 'k', 'time', 'conv', 'correct']

grouped = df.groupby(['n', 'e'], as_index=False).mean()
ns = grouped['n'].unique()

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.grid()

for n in ns:
    temp_df = grouped.loc[grouped['n'] == n]
    _x = temp_df['n'].dropna().tolist()
    _y = temp_df['e'].dropna().tolist()
    _z = temp_df['k'].dropna().tolist()
    ax.plot3D(_x, _y, _z)

ax.set_title(fr'Number of iterations required for convergence')
ax.elev = 0
ax.azim = 0
# Set axes label
ax.set_xlabel('Number of nodes', labelpad=20)
ax.set_ylabel('Number of edges', labelpad=20)
ax.set_zlabel('k', labelpad=20)
plt.show()
print()

print(grouped['k'].mean())
print(grouped['conv'].mean())
print(grouped['correct'].mean())
