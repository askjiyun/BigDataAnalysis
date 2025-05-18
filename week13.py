from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
# Logistic Regression
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(max_iter=1000).fit(X, y)
print(clf.predict(X[:2, :]))
print(clf.predict_proba(X[:2, :]))
print(clf.score(X, y))

print("----------------------------------------------")

# MLP 다층 퍼셉트론 분류
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=2000, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=1)
clf = MLPClassifier(max_iter=300).fit(X_train, y_train)
print(clf.predict_proba(X_test[:1]))
print(clf.predict(X_test[:5, :]))
print(clf.score(X_test, y_test))

print("------------------------------")

# MLP 다층 퍼셉트론 회귀
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
X, y = make_regression(n_samples=5000, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                     random_state=1)
regr = MLPRegressor(max_iter=500).fit(X_train, y_train)
print(regr.predict(X_test[:2]))
print(regr.score(X_test, y_test))



