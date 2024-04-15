# Dataset:
# year,degree,age
# 3,M,18
# NaN,B,18
# 2,M,21
# NaN,B,19
# NaN,B,19
# 2,NaN,29
# 1,NaN,28
# 3,M,29
# 1,M,28
# 3,NaN,18
# 1,NaN,27
# 2,M,25
# 2,M,20
# 1,B,20
# 3,B,23

import numpy as np
# build pandas dataframe
import pandas as pd

data = {
    'year': [3, np.nan, 2, np.nan, np.nan, 2, 1, 3, 1, 3, 1, 2, 2, 1, 3],
    'degree': ['M', 'B', 'M', 'B', 'B', np.nan, np.nan, 'M', 'M', np.nan, np.nan, 'M', 'M', 'B', 'B'],
    'age': [18, 18, 21, 19, 19, 29, 28, 29, 28, 18, 27, 25, 20, 20, 23]
}
df = pd.read_csv('../../data/hyperskill-dataset-96829169.txt')
# Calculate the proportions of missing values per column using pandas methods.
proportions = df.isnull().mean()
# Round the values to the second decimal place.
proportions = proportions.round(2)
# Print the result.
print(proportions)