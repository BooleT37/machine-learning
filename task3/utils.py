import numpy as np


def count_false_positives_and_negatives(actual, forecast):
    false_positives = np.logical_and(forecast == 1, actual == 0)
    false_negatives = np.logical_and(forecast == 0, actual == 1)
    return false_positives, false_negatives
