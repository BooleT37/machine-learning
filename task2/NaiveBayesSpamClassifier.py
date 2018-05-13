import math
from functools import reduce

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from task2.utils import combined_content

TOKEN_PATTERN = r'(?u)\b[a-zA-Z_]{2,}\b'


# https://ru.wikipedia.org/wiki/%D0%91%D0%B0%D0%B9%D0%B5%D1%81%D0%BE%D0%B2%D1%81%D0%BA%D0%B0%D1%8F_%D1%84%D0%B8%D0%BB%D1%8C%D1%82%D1%80%D0%B0%D1%86%D0%B8%D1%8F_%D1%81%D0%BF%D0%B0%D0%BC%D0%B0
class NaiveBayesSpamClassifier:
    def __init__(self):
        self.bag_total = {}
        self.spam_num = 0
        self.ham_num = 0
        self.vect_spam = None
        self.vect_ham = None
        self.sm_spam = None
        self.sm_ham = None

    def fit(self, files_spam, files_ham):
        self.vect_spam = CountVectorizer(token_pattern=TOKEN_PATTERN, binary=True)
        self.vect_ham = CountVectorizer(token_pattern=TOKEN_PATTERN, binary=True)

        # Spam and ham words sparse matrix
        self.sm_spam = self.vect_spam.fit_transform(combined_content(files_spam))
        self.sm_ham = self.vect_ham.fit_transform(combined_content(files_ham))

        # Spam and ham messages number
        self.spam_num = len(files_spam)
        self.ham_num = len(files_ham)

    # freq of spam messages with given word per all spam messages (Pr(W|S))
    def prob_word_in_spam(self, token):
        voc = self.vect_spam.vocabulary_
        if token in voc:
            return self.sm_spam[:, voc[token]].sum() / self.spam_num
        if token in self.vect_ham.vocabulary_:
            return 0
        return None

    # freq of ham messages with given word per all ham messages (Pr(W|H))
    def prob_word_in_ham(self, token):
        voc = self.vect_ham.vocabulary_
        if token in voc:
            return self.sm_ham[:, voc[token]].sum() / self.ham_num
        if token in self.vect_spam.vocabulary_:
            return 0
        return None

    def predict_text(self, text):
        tokenizer = self.vect_spam.build_tokenizer()
        pp = []
        for token in tokenizer(text):
            prob_word_in_spam = self.prob_word_in_spam(token)
            prob_word_in_ham = self.prob_word_in_ham(token)
            if prob_word_in_spam is not None:
                # Cond probability of message being spam given word W (Pr(S|W))
                pp.append(prob_word_in_spam / (prob_word_in_spam + prob_word_in_ham))

        # using log to prevent underflows
        # https://stats.stackexchange.com/questions/105602/example-of-how-the-log-sum-exp-trick-works-in-naive-bayes/253319#253319
        pp_sum_log_spam = reduce(lambda v, p: v + (0 if p == 0 else math.log(p)), pp, 0)
        pp_sum_log_ham = reduce(lambda v, p: v + (0 if p == 1 else math.log(1 - p)), pp, 0)
        return 1 if pp_sum_log_spam > pp_sum_log_ham else 0

    def predict(self, filenames):
        df = pd.DataFrame(columns=['name', 'is_spam'])
        for i, filename in enumerate(filenames):
            content = open(filename).read()
            df = df.append({'name': filename, 'is_spam': self.predict_text(content)}, ignore_index=True)

        return df
