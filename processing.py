import pandas as pd

beta = 2
df = pd.read_csv(f'experiments/dense_beta_{beta}.csv')
df.columns = ['n', 'e', 'k', 'time', 'conv', 'correct']

grouped = df.groupby(['n', 'e'], as_index=False).mean()
ns = grouped['n'].unique()

for n in ns:
    temp_df = grouped.loc[grouped['n'] == n]
    print(temp_df)