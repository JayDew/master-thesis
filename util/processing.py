import pandas as pd

alpha = 0.1
beta = 1
df = pd.read_csv(f'../plaintext/pgd.csv')
df.columns = ['n', 'e', 'k', 'time', 'conv', 'correct', 'error']

grouped = df.groupby(['n', 'e'], as_index=False).mean()
ns = grouped['n'].unique()

for n in ns:
    temp_df = grouped.loc[grouped['n'] == n]
    print(temp_df[['n', 'e', 'k', 'conv', 'correct', 'error']])