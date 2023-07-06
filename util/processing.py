import pandas as pd

df = pd.read_csv(f'../experiments/final/c_100/alpha_0.00001/apgd.csv')
df.columns = ['n', 'e', 'k_correct', 'k', 'time', 'conv', 'correct', 'error']

grouped = df.groupby(['n', 'e'], as_index=False).mean()
ns = grouped['n'].unique()

for n in ns:
    temp_df = grouped.loc[grouped['n'] == n]
    print(temp_df[['n', 'e', 'k', 'correct', 'error', 'time']])