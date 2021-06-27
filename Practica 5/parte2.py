import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("parte2.tsv", sep=',').drop(["day"], axis=1)
ord_enc = OrdinalEncoder()


def Map(df):
    if 'beach' in df.columns:
        df['beach'] = df.beach.map({'No': 0, 'Yes': 1})
    df['humidity'] = df.humidity.map({'Normal': 0, 'High': 1})
    df['temp'] = df.temp.map({'Low': 0, 'Mild': 1, 'High': 2})
    df['outlook'] = df.outlook.map({'Rain': 0, 'Cloudy': 1, 'Sunny': 2})
    return df


df = Map(df)

X = df.drop(["beach"], axis=1)
y = df['beach']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25)

# train
model = GaussianNB()
model.fit(X_train, y_train)

# test
print(model.score(X_test, y_test))

# Create the pandas DataFrame
instances = pd.DataFrame([['Sunny', 'Mild', 'High'], ['Cloudy', 'Low', 'High'], ['Rain', 'High', 'Normal']],
                  columns=['outlook', 'temp', 'humidity'])

instances = Map(instances)

prediction = model.predict(instances)
probabilities = model.predict_proba(instances)

mat = confusion_matrix(y_test, prediction)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.title("MultinomialNB")
plt.show()



print()






