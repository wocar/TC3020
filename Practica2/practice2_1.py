import sys

from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn import tree
import william as w
from sklearn.preprocessing import StandardScaler
from pandas import *

def practica(dataset, lr=0.05, iterations=3500):
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target,
                                                        test_size=0.2, shuffle=True, random_state=123)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    print(f"Decision Tree Classifier score: {score}%")

    model = LogisticRegression(verbose=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    print(f"Logistic Regression results (Ski-Learn): {score}%")

    model = w.LogisticRegression(lr, iterations)
    theta = X_train.shape[1] * [1]
    model.fit(X_train, y_train, theta)
    y_pred = model.predict(X_test, threshold=.5)

    print(f"Self implementation results: {w.report(y_pred, y_test)}")

print("Iris dataset: ")
print("-"*10)
practica(load_iris())


print("Breast cancer dataset: ")
print("-"*10)
practica(load_breast_cancer())

print("Wine dataset: ")
print("-"*10)
practica(load_wine(), 0.001, 3000)


