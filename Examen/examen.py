import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def examen(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    print(f"Logistic regression: {model.score(X_test, y_test)}")
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    print(f"Tree classifier: {model.score(X_test, y_test)}")
    model = KNeighborsClassifier(n_neighbors=50)
    model.fit(X_train, y_train)
    print(f"50-NN: {model.score(X_test, y_test)}")


print(f"Glass dataset")
print(f"-" * 20)

glass = pd.read_csv('glass.txt', sep=' ').drop('ID', axis=1)
X = glass.drop('Type', axis=1)
y = glass["Type"]

examen(X, y)

print(f"\n\nJohns Hopkins University Ionosphere database")
print(f"-" * 20)
glass = pd.read_csv('ion.txt', sep=' ').drop('ID', axis=1).drop('V1', axis=1).drop('V2', axis=1)

X = glass.drop('Class', axis=1)
y = glass["Class"]

examen(X, y)


# P = 10
