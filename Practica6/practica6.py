from itertools import cycle

from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, roc_auc_score, plot_roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    test_size=0.20, random_state=123)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


def test(model, name):
    model.fit(X_train, y_train)
    y_hat_test = model.predict(X_test)
    mat = confusion_matrix(y_test, y_hat_test)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.title(name + "\nPrecision: " + str(model.score(X_test, y_test)))
    plt.show()


test(LogisticRegression(max_iter=200, multi_class='ovr'), "LogisticRegression(multi_class='ovr')")
test(KNeighborsClassifier(), "KNeighborsClassifier")
test(svm.SVC(kernel='linear', C=1, decision_function_shape='ovo'),
     "svm.SVC(kernel='linear', C=1, decision_function_shape='ovo')")
test(svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo'),
     "svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo')")
test(svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo'),
     "svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo')")
test(svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo'),
     "svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo')")
