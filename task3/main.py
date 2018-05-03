from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB

from task3.datareader import read_csv
from task3.utils import count_false_positives_and_negatives

TOTAL_VALUES_COUNT = 10000
TRAIN_VALUES_PART = 0.8
TRAIN_VALUES_COUNT = int(TOTAL_VALUES_COUNT * TRAIN_VALUES_PART)

df_train_test = read_csv('./data/train.csv', True, TOTAL_VALUES_COUNT)
df_train = df_train_test[:TRAIN_VALUES_COUNT]
df_test = df_train_test[TRAIN_VALUES_COUNT:]

vectorizer = HashingVectorizer(alternate_sign=False)
X_train = vectorizer.transform(df_train['name'])
X_test = vectorizer.transform(df_test['name'])

y_train = df_train['isOrg'].map(lambda v: 1 if v else 0)
y_test = df_test['isOrg'].map(lambda v: 1 if v else 0)


clf = MultinomialNB(alpha=.005)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


score = metrics.precision_score(y_test, y_pred)
false_positives, false_negatives = count_false_positives_and_negatives(y_test, y_pred)
false_positives_str = '"' + '", "'.join(df_test["name"][false_positives][:3]) + '"'
false_positives_sum = sum(false_positives)
if false_positives_sum > 3:
    false_positives_str += ", ..."

false_negatives_str = '"' + '", "'.join(df_test["name"][false_negatives][:3]) + '"'
false_negatives_sum = sum(false_negatives)
if false_negatives_sum > 3:
    false_negatives_str += ", ..."

print(f'Out of a total {len(df_train_test)} points: \n'
      f'\tfalse positives: {false_positives_sum} ({false_positives_str})\n'
      f'\tfalse negatives: {false_negatives_sum} ({false_negatives_str})\n'
      f'\tprecision: {score}')