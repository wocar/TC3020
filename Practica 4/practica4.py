import random
from math import nan

from numpy import NaN
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import pandas
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from Practica3.william import load_default_dataset

# Punto #1
df = load_default_dataset(as_frame=True)
X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=1)

X_train_original = X_train.copy(deep=True)
# Punto #2
for index, row in X_train.iterrows():
    deleteBalanceAndIncome = random.choices([True, False], [.2, .8])[0]
    if deleteBalanceAndIncome:
        X_train.at[index, 'balance'] = None
        X_train.at[index, 'income'] = None

    deleteBalance = random.choices([True, False], [.1, .9])[0]
    if deleteBalance:
        X_train.at[index, "balance"] = None

# Punto 3 y 4

# Punto 4 a)

X_train_T1 = X_train.copy(deep=True)
meanBalance = X_train["balance"].mean()
meanIncome = X_train["income"].mean()
for index, row in X_train_T1.iterrows():
    if np.isnan(row["balance"]):
        X_train_T1.at[index, 'balance'] = meanBalance
    if np.isnan(row["income"]):
        X_train_T1.at[index, 'income'] = meanIncome

dataset_balance = X_train.copy(deep=True).join(y_train)
X_train_balance = df.drop("balance", axis=1)
y_train_balance = df["balance"]

balanceModel = LinearRegression()
balanceModel.fit(X_train_balance, y_train_balance)

dataset_income = dataset_balance.copy(deep=True)
X_train_income = df.drop("income", axis=1)
y_train_income = df["income"]

incomeModel = LinearRegression()
incomeModel.fit(X_train_income, y_train_income)

X_train_T2 = X_train.copy(deep=True)

for index, row in X_train_T2.iterrows():
    if np.isnan(row["balance"]):
        X_train_T2.at[index, 'balance'] = balanceModel.predict([X_train_balance.loc[index]])
    if np.isnan(row["income"]):
        X_train_T2.at[index, 'income'] = incomeModel.predict([X_train_income.loc[index]])

X_train_T3 = X_train.copy(deep=True)
knnIncomeModel = KNeighborsRegressor()
knnIncomeModel.fit(X_train_income, y_train_income)
knnBalanceModel = KNeighborsRegressor()
knnBalanceModel.fit(X_train_balance, y_train_balance)

for index, row in X_train_T3.iterrows():
    if np.isnan(row["balance"]):
        X_train_T3.at[index, 'balance'] = knnBalanceModel.predict([X_train_balance.loc[index]])
    if np.isnan(row["income"]):
        X_train_T3.at[index, 'income'] = knnIncomeModel.predict([X_train_income.loc[index]])

# Punto #5


model = LogisticRegression()
model.fit(X_train_original, y_train)
print(f"Logistic regression score T original: {model.score(X_test, y_test)}")

model = LogisticRegression()
model.fit(X_train_T1, y_train)
print(f"Logistic regression score T'1: {model.score(X_test, y_test)}")

model = LogisticRegression()
model.fit(X_train_T2, y_train)
print(f"Logistic regression score T'2: {model.score(X_test, y_test)}")

model = LogisticRegression()
model.fit(X_train_T3, y_train)
print(f"Logistic regression score T'3 : {model.score(X_test, y_test)}")

model = DecisionTreeClassifier()
model.fit(X_train_original, y_train)
print(f"Tree clf score T original: {model.score(X_test, y_test)}")

model = DecisionTreeClassifier()
model.fit(X_train_T1, y_train)
print(f"Tree clf score T'1: {model.score(X_test, y_test)}")

model = DecisionTreeClassifier()
model.fit(X_train_T2, y_train)
print(f"Tree clf score T'2: {model.score(X_test, y_test)}")

model = DecisionTreeClassifier()
model.fit(X_train_T3, y_train)
print(f"Tree clf score T'3 : {model.score(X_test, y_test)}")


model = KNeighborsClassifier(n_neighbors=50)
model.fit(X_train_original, y_train)
print(f"KNN score T original: {model.score(X_test, y_test)}")

model = KNeighborsClassifier(n_neighbors=50)
model.fit(X_train_T1, y_train)
print(f"KNN score T'1: {model.score(X_test, y_test)}")

model = KNeighborsClassifier(n_neighbors=50)
model.fit(X_train_T2, y_train)
print(f"KNN score T'2: {model.score(X_test, y_test)}")

model = KNeighborsClassifier(n_neighbors=50)
model.fit(X_train_T3, y_train)
print(f"KNN score T'3 : {model.score(X_test, y_test)}")


