import math

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target
# Leave the last 50 observations from the dataset for a test set.

X_train = X[:-50]
X_test = X[-50:]

# Use the X_train set to build a Linear Regression model.

# Create a Linear Regression model
model = LinearRegression()

# Fit the model with the training data
model.fit(X_train, y[:-50])

# Use the model to predict the y values from the X_test set
predictions = model.predict(X_test)

# What is the MSE score of the resulting model on the test data?
print(math.ceil(mean_squared_error(y[-50:], predictions)))