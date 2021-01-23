from batchedmoments import BatchedMoments


def test_batches():
    data = [list(range(100))] * 10
    full = BatchedMoments()(data)
    reduced = BatchedMoments(axis=1)(data).reduce(0)
    assert full == reduced
