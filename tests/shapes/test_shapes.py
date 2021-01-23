from random import randint

from batchedmoments import BatchedMoments


def test_shape_None():
    bm = BatchedMoments(shape=None)
    assert bm._initialized is False  # first update call will initialize
    assert bm._moments_shape is None  # moments shape is zero


def test_shape_0d():
    shp = tuple()
    bm = BatchedMoments(shape=shp)
    assert bm._initialized is True
    assert bm._moments_shape == shp


def test_shape_random_nd():
    shp = tuple([4] * randint(1, 16))
    bm = BatchedMoments(shape=shp)
    assert bm._initialized is True
    assert bm._moments_shape == shp
