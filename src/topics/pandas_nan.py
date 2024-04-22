import pandas as pd

df = pd.read_csv('../../data/hyperskill-dataset-97148377.txt', sep=',')
df.head()
df.info()
print(df.describe())
print(df['Age'].max())

df = pd.read_csv('../../data/hyperskill-dataset-97148533.txt', index_col='Name')
print(df.head(n=10))
