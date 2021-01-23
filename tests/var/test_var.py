import numpy as np
from scipy.stats import tvar, norm

from batchedmoments import BatchedMoments


def test_correctness():
    data = norm.rvs(size=1000, random_state=3)  # mean = 0.01728433
    bm = BatchedMoments(axis=0)(data)
    assert np.allclose(tvar(data, ddof=0), bm.variance, equal_nan=True)
