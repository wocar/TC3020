import numpy as np
import os

import pandas
from numpy import genfromtxt, random
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import Bunch
import pandas as pd
import math
from joblib import Parallel, delayed
import multiprocessing
from collections import Counter
import itertools

class WilliamLogisticRegression:
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

    def fit(self, X, y, theta=None):

        if theta is None:
            theta = np.random.rand(X.shape[1])

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

    def predict(self, X, threshold=0.5):
        return self.__sigmoid(self.predict_prob(X)) >= threshold

    def report(self, y_pred, y_test):

        c = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                c += 1

        rate = c / len(y_pred)

        return rate


def sci_to_pd(bunch):
    return pd.DataFrame(data=np.c_[bunch['data'], bunch['target']],
                        columns=bunch['feature_names'] + ['target'])


def load_default_dataset(as_frame=False):
    def convert_bool(col):
        if str(col).title() == "Yes":
            return 1
        elif str(col).title() == "No":
            return 0
        else:
            return col

    frame = pandas.read_csv('default.txt', delimiter='\t', converters={"default": lambda x: convert_bool(x),
                                                                       "student": lambda x: convert_bool(
                                                                           x)})

    if as_frame:
        return frame

    dataset = frame.to_numpy()
    data = dataset[:, -3:]  # Features
    feature_names = frame.columns[-3:]  # Features

    target = dataset[:, 1]  # Target
    target_names = ['default']  # Features

    return Bunch(data=data,
                 target=target,
                 frame=None,
                 target_names=target_names,
                 DESCR='default.txt',
                 feature_names=feature_names,
                 filename='default.txt')


def load_gender_dataset(as_frame=False):
    def convert_bool(col):
        if str(col).title() == "Male":
            return 1
        elif str(col).title() == "Female":
            return 0
        else:
            return col

    frame = pandas.read_csv('genero.txt', delimiter=',', converters={"Gender": lambda x: convert_bool(x)})

    if as_frame:
        return frame

    dataset = frame.to_numpy()
    data = dataset[:, -2:]  # Features
    feature_names = frame.columns[-2:]  # Features

    target = dataset[:, 1]  # Target
    target_names = ['Gender']  # Features

    return Bunch(data=data,
                 target=target,
                 frame=None,
                 target_names=target_names,
                 DESCR='genero.txt',
                 feature_names=feature_names,
                 filename='genero.txt')


class WilliamKNN:
    # p = 1 (Manhattan)
    # p = 2 (Euclidean)
    def __init__(self):
        self.total = 0
        self.progress = 0

    @staticmethod
    def minkowski_distance(a, b, p=1):
        dim = len(a)
        distance = 0

        # Calculate minkowski distance using parameter p
        for d in range(dim):
            distance += abs(a[d] - b[d]) ** p
        distance = distance ** (1 / p)

        return distance

    def knn_predict(self, X_train, X_test, y_train, k, p):

        # Counter to help with label voting

        # Make predictions on the test data
        # Need output of 1 prediction per test data point
        y_train = y_train.reset_index(drop=True)
        self.total = len(X_train) * len(X_test)

        num_cores = multiprocessing.cpu_count()

        X_test_split = self.chunk(X_test, num_cores)

        results = Parallel(n_jobs=num_cores)(
            delayed(self.process_point)(ix, X_test_split[ix], X_train, y_train, k,p ) for ix in range(len(X_test_split)))

        results = np.array(results, dtype=object)[:,1]
        return list(itertools.chain.from_iterable(results))

        print("")

    def process_point(self, ix, X_test, X_train, y_train,k,p ):
        y_hat_test = []

        j = 0
        for test_point in X_test:
            i = 0
            distances = []
            j += 1

            for train_point in X_train:
                distance = self.minkowski_distance(test_point, train_point, p=p)
                distances.append([distance, i])
                self.progress += 1 # Can cause a race condition, won't use lock to increase perf
                i += 1
                if i % 1000 == 0:
                    distances.sort(key=lambda x: x[0])
                    distances = distances[:k + 1]

            if j % 30 == 0 and ix == 0:
                print(f"Progress: {(self.progress / self.total) * 100}%")

            distances.sort(key=lambda x: x[0])
            distances = np.array(distances[:k])

            counter = Counter(y_train[distances[:, 1]])

            prediction = counter.most_common()[0][0]

            y_hat_test.append(prediction)
        return ix, y_hat_test

    def flatmap(self, func, *iterable):
        return itertools.chain.from_iterable(map(func, *iterable))

    def chunk(self, seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out
# References:
# - https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62
# - https://github.com/arseniyturin/logistic-regression
# - https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
# - https://medium.com/@lope.ai/logistic-regression-from-scratch-in-python-d1e9dd51f85d
# - https://towardsdatascience.com/how-to-build-knn-from-scratch-in-python-5e22b8920bd2

# - References
