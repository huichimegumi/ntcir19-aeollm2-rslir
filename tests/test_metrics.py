import numpy as np

from aeollm2_osodt.metrics import pairwise_accuracy, weighted_total


def test_pairwise_accuracy_handles_ties_exactly():
    gold = np.array([3.0, 2.0, 2.0])
    pred = np.array([5.0, 4.0, 4.0])
    accuracy, correct, total = pairwise_accuracy(gold, pred)
    assert (accuracy, correct, total) == (1.0, 3, 3)


def test_weighted_total_normalizes_dimension_weights():
    values = np.array([[10.0, 0.0, 0.0, 0.0]])
    weights = {
        "comprehensiveness": 2.0,
        "insight": 1.0,
        "instruction_following": 1.0,
        "readability": 0.0,
    }
    assert weighted_total(values, weights)[0] == 5.0
