from random import randint

from batchedmoments import BatchedMoments


def test_axis_None():
    bm = BatchedMoments(axis=None)
    assert bm._initialized is False  # first update call will initialize
    assert bm._moments_shape is None  # moments shape is zero


def test_axis_0d():
    shp = tuple()
    bm = BatchedMoments(axis=shp)
    assert bm._initialized is False
    assert bm.axis == shp


def test_axis_random_nd():
    shp = tuple([4] * randint(1, 16))
    bm = BatchedMoments(axis=shp)
    assert bm._initialized is False
    assert bm.axis == shp
