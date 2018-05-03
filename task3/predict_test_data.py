import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB

from task3.datareader import read_csv


df_train = read_csv('./data/train.csv', True)
df_test = read_csv('./data/test.csv', False)

vectorizer = HashingVectorizer(alternate_sign=False)
X_train = vectorizer.transform(df_train['name'])
X_test = vectorizer.transform(df_test['name'])

y_train = df_train['isOrg'].map(lambda v: 1 if v else 0)


clf = MultinomialNB(alpha=.005)
clf.fit(X_train, y_train)
y_pred = pd.Series(clf.predict(X_test))

df_test['isOrg'] = y_pred.map(lambda v: "True" if v == 1 else "False")
df_test.to_csv('data/test_predicted.csv', sep=' ')