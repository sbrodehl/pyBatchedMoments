import numpy as np
from scipy.stats import skew

from batchedmoments import BatchedMoments


def test_correctness_1():
    data = np.array([1, 2, 3, 4, 5])  # skewness = 0.0
    bm = BatchedMoments(axis=0)(data)
    assert np.allclose(skew(data), bm.skewness, equal_nan=True)


def test_correctness_2():
    data = np.array([2, 8, 0, 4, 1, 9, 9, 0])  # skewness = 0.2650554122698573
    bm = BatchedMoments(axis=0)(data)
    assert np.allclose(skew(data), bm.skewness, equal_nan=True)
