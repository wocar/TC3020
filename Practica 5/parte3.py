import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB  # Gaussian naive Bayes classifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

sc = StandardScaler()

data = load_iris()
iris = pd.DataFrame(data.data, columns=data.feature_names)


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
    test_size=0.25, random_state = 123)

#train
model = GaussianNB()
model.fit(X_train, y_train)

#test
predicted = model.predict(X_test)
print("GaussianNB")
print(model.score(X_test, y_test))
print(confusion_matrix(y_test, predicted))
print(classification_report(y_test, predicted))

mat = confusion_matrix(y_test, predicted)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.title("GaussianNB")
plt.show()




#plt.scatter(predicted,color='red')
#plt.show()





#train
model = MultinomialNB()
model.fit(X_train, y_train)

#test
predicted = model.predict(X_test)
print("MultinomialNB")
print(model.score(X_test, y_test))
print(confusion_matrix(y_test, predicted))
print(classification_report(y_test, predicted))

mat = confusion_matrix(y_test, predicted)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.title("MultinomialNB")
plt.show()



print()
