from random import randint, sample

from batchedmoments import BatchedMoments


def test_axis_shape_None():
    bm = BatchedMoments(axis=None, shape=None)
    assert bm._initialized is False  # first update call will initialize
    assert bm.shape is None  # moments shape is None
    assert bm.axis is None  # axis shape is None


def test_axis_shape_0d():
    shp = tuple()
    bm = BatchedMoments(axis=shp, shape=shp)
    assert bm._initialized is True
    assert bm.shape == shp
    assert bm.axis == shp


def test_axis_shape_random_nd():
    shp = tuple([4] * randint(1, 6))
    axs = tuple(sample(range(len(shp)), randint(1, len(shp))))
    bm = BatchedMoments(axis=axs, shape=shp)
    assert bm._initialized is True
    assert bm.shape == shp
    assert bm.axis == axs
