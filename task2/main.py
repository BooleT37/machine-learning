import os
import pandas as pd
import numpy as np

from task2.NaiveBayesSpamClassifier import NaiveBayesSpamClassifier

TRAIN_VALUES_PART = 0.99

files_spam = list(map(lambda fname: 'data/spam/' + fname, os.listdir("data/spam")))
files_ham = list(map(lambda fname: 'data/notSpam/' + fname, os.listdir("data/notSpam")))


def validate():
    num_files_spam_train = int(TRAIN_VALUES_PART * len(files_spam))
    num_files_ham_train = int(TRAIN_VALUES_PART * len(files_ham))

    files_spam_train = files_spam[:num_files_spam_train]
    files_ham_train = files_ham[:num_files_ham_train]

    files_spam_validating = files_spam[num_files_spam_train:]
    files_ham_validating = files_ham[num_files_ham_train:]

    num_files_spam_vd = len(files_spam_validating)
    num_files_ham_vd = len(files_ham_validating)
    num_files_vd = num_files_spam_vd + num_files_ham_vd

    df_vd_spam = pd.DataFrame({'name': files_spam_validating, 'is_spam': [1] * num_files_spam_vd})
    df_vd_ham = pd.DataFrame({'name': files_ham_validating, 'is_spam': [0] * num_files_ham_vd})
    df_vd = (df_vd_spam.append(df_vd_ham)).sample(frac=1)

    classifier = NaiveBayesSpamClassifier()

    classifier.fit(files_spam_train, files_ham_train)
    df_pred = classifier.predict(df_vd['name'])

    false_positives = df_pred['name'][np.logical_and(df_pred['is_spam'] == 1, df_vd['is_spam'] == 0)]
    false_negatives = df_pred['name'][np.logical_and(df_pred['is_spam'] == 0, df_vd['is_spam'] == 1)]

    num_true_positives = np.sum(np.logical_and(df_pred['is_spam'] == 1, df_vd['is_spam'] == 1))
    num_true_negatives = np.sum(np.logical_and(df_pred['is_spam'] == 0, df_vd['is_spam'] == 0))

    accuracy = (num_true_positives + num_true_negatives) / num_files_vd

    print(f'accuracy: {accuracy} ({num_true_positives + num_true_negatives} / {num_files_vd})')
    if len(false_positives) != 0:
        print(f'False positives [{len(false_positives)}]:\n\t' + '\n\t'.join(false_positives))
    if len(false_negatives) != 0:
        print(f'False negatives [{len(false_negatives)}]:\n\t' + '\n\t'.join(false_negatives))


def predict():
    files_test = list(map(lambda fname: 'data/unknown/' + fname, os.listdir("data/unknown")))

    classifier = NaiveBayesSpamClassifier()
    classifier.fit(files_spam, files_ham)
    df_pred = classifier.predict(files_test)
    df_pred.to_csv('bayes_result.csv', index=False, sep=';')


# validate()
predict()
