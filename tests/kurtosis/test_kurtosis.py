import numpy as np
from scipy.stats import kurtosis, norm

from batchedmoments import BatchedMoments


def test_correctness():
    data = norm.rvs(size=1000, random_state=3)  # kurtosis = -0.06928694200380558
    bm = BatchedMoments(axis=0)(data)
    assert np.allclose(kurtosis(data), bm.kurtosis, equal_nan=True)
