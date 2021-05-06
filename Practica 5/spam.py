import string
import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from nltk.corpus import stopwords

sw = stopwords.words("english")

# load data
df = pd.read_csv('spam.csv',
                 usecols=["v1", "v2"])
df.columns = ["label", "message"]

# normalize

df['label'] = df.label.map({'ham': 0, 'spam': 1})
df['message'] = df.message.map(lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower())    # Sin normalizar

df['message'] = df['message'].apply(nltk.word_tokenize)
df['message'] = df.message.map(lambda x: [i for i in x if i not in sw]) # Sin normalizar

stemmer = PorterStemmer()
df['message'] = df['message'].apply(lambda x: [stemmer.stem(y) for y in x])  # Sin normalizar
df['message'] = df['message'].apply(lambda x: ' '.join(x))

count_vect = CountVectorizer()
counts = count_vect.fit_transform(df['message'])
transformer = TfidfTransformer().fit(counts)    # Sin normalizar
counts = transformer.transform(counts)  # Sin normalizar

# training


X_train, X_test, y_train, y_test = \
    train_test_split(counts, df['label'], test_size=0.2, random_state=2)

model = MultinomialNB().fit(X_train, y_train)

# test
predicted = model.predict(X_test)
print(model.score(X_test, y_test))
print(confusion_matrix(y_test, predicted))
print(classification_report(y_test, predicted))

print("")
