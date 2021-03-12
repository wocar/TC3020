import numpy as np
import os
from numpy import genfromtxt, random
from sklearn.metrics import confusion_matrix, classification_report


class LogisticRegression:
    def __init__(self, lr=0.01, iterations=1000, fit_intercept=False, verbose=False):
        self.lr = lr
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.iterations = iterations

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y, theta):

        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weights initialization
        self.theta = theta
        self.loss = []

        size = y.size
        i = 0
        for _ in range(self.iterations):
            h = self.__sigmoid(np.dot(X, self.theta))
            gradient = np.dot(X.T, (h - y)) / size

            anterior = self.theta

            self.theta = self.theta - (self.lr * gradient)
            delta = np.abs((anterior - self.theta)).mean()

            if self.verbose and i % 1000 == 0:
                print(
                    f'Threshold reached: iterations: {i} delta: {delta} loss: {self.__loss(h, y)} beta: {self.theta} lr = {self.lr}\n\n')

            if (np.abs(delta) < 0.000001):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(
                    f'Threshold reached: iterations: {i} delta: {delta} loss: {self.__loss(h, y)} beta: {self.theta} lr = {self.lr}\n\n')
                break

            i += 1

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return (np.dot(X, self.theta))

    def predict(self, X, threshold):
        return self.__sigmoid(self.predict_prob(X)) >= threshold


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


def load_data_set(dataset, target_x_columns, target_y_column, percentage_test=.8):
    data = genfromtxt(dataset, delimiter=',', skip_header=1)
    random.shuffle(data)

    X_data = data[:, target_x_columns:]  # Features
    y_data = data[:, target_y_column]  # Target

    ochenta = int(len(data) * percentage_test)

    X_train = X_data[:ochenta]
    X_test = X_data[ochenta:]

    y_train = y_data[:ochenta]
    y_test = y_data[ochenta:]

    print(f"Loaded dataset '{dataset}' Training: {len(X_train)}\t\tTest: {len(X_test)}")
    return X_train, y_train, X_test, y_test


def report(y_pred, y_test):

    c = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            c += 1

    rate = c / len(y_pred)

    return rate

# References:
# - https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62
# - https://github.com/arseniyturin/logistic-regression
# - https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
# - https://medium.com/@lope.ai/logistic-regression-from-scratch-in-python-d1e9dd51f85d
