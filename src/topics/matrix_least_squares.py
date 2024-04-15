import numpy as np

# hyperskill problem https://hyperskill.org/repeat/step/36199
# least squares solution

X = np.array([[1,1,1,1],[-1,0,1,3],[2,1,3,-2],[4,3,0,-1]])
y = np.array([0,3,1,-2])

# pseudo-inverse of X
X_dag = np.linalg.pinv(X)
beta = X_dag.dot(y)
# best approximation of y
y_hat = X.dot(beta)
print(y_hat)


