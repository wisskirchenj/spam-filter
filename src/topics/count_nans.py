# read in a csv file into a pandas dataframe
from statistics import mode, median

import pandas as pd
from numpy import mean

df = pd.read_csv('../../data/hs.csv', encoding='iso-8859-1')

# count the number of columns in the dataframe, that contain NaN values
# and print the result

print(df.isna())
print(df.isna().any())
print(df.isna().any().sum())

a =[5, 7, 8, 9, 10, 12, 14, 15, 16, 20]
print(mode(a))
print("Med", median(a))
aquer = mean(a)
sum = 0
for i in a:
    sum += abs(i - aquer)
print(sum)
print(sum/len(a))


