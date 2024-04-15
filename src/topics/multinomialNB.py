# https://hyperskill.org/learn/step/34351
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X, y = load_iris(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

model = MultinomialNB()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))