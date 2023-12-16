# create ndarray from these data:
# 4	2
# 6	2
# 8	3
# 10	5
# 12	5
# 14	6
# 16	6

import numpy as np
from sklearn.linear_model import LinearRegression

arr = np.array([[4, 2], [6, 2], [8, 3], [10, 5], [12, 5], [14, 6], [16, 6]])
# X = first column

X = np.array(arr[:,0]).reshape(-1,1)

# y = second column
Y = arr[:,1]
print(X)
print(Y)
model = LinearRegression()
model.fit(X, Y)
print(model.predict([[23]]))