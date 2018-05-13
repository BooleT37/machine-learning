import os
from task2.NaiveBayesSpamClassifier import NaiveBayesSpamClassifier


files_spam = list(map(lambda fname: 'data/spam/' + fname, os.listdir("data/spam")))
files_ham = list(map(lambda fname: 'data/notSpam/' + fname, os.listdir("data/notSpam")))
files_test = list(map(lambda fname: 'data/unknown/' + fname, os.listdir("data/unknown")))

classifier = NaiveBayesSpamClassifier()
classifier.fit(files_spam, files_ham)
df_pred = classifier.predict(files_test)
df_pred.to_csv('bayes_result.csv', index=False, sep=';')
