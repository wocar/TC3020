#!/usr/bin/env python
from numpy import random
import warnings
import numpy as np
import william as w
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

X_train, y_train, X_test, y_test = w.load_data_set("default_clean.txt", -3, 1)


print(f"Training with william's LogisticRegression implementation.... ")

model = w.LogisticRegression(lr=0.00000005, verbose=False)
theta = np.random.rand(X_train.shape[1])
model.fit(X_train, y_train, theta)
y_pred = model.predict(X_test, threshold=.5)

print(f"Test results: ")

w.report(y_pred, y_test)

print(f"Training with SciLearn's LogisticRegression implementation.... ")

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Test results: ")

w.report(y_pred, y_test)